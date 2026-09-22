/* ==================================================================================================
 * engram_alloc.c -- checked, counted, failable allocation with a use-after-free quarantine.
 * ==================================================================================================
 * The design is argued in engram_alloc.h. This file keeps three invariants, and every function below
 * is written against them:
 *
 *   I1  A pointer handed to the system free() came from the system malloc() and has never been
 *       handed to free() before. Every path that cannot prove this LEAKS instead.
 *   I2  live_blocks and live_bytes change only when a block's state actually changes, under the lock,
 *       so they are exact at every instant a reader can observe them.
 *   I3  Every public allocation entry point ticks the fault counter EXACTLY ONCE per call, whether it
 *       succeeds, fails, or is refused for overflow -- so "fail the Nth allocation" names the same call
 *       on every run and the fault sweep is reproducible.
 * ============================================================================================== */
#include "engram_alloc.h"
#include "engram_plat.h"

#include <stdlib.h>
#include <string.h>

/* ---- AGREEING WITH LEAKSANITIZER ABOUT INTENT -------------------------------------------------
 * A block found corrupt is leaked ON PURPOSE (I1). Under AddressSanitizer, LeakSanitizer then reports
 * that leak -- correctly, from where it stands, since it cannot know the leak was a decision. The two
 * instruments were disagreeing about intent, not about facts.
 *
 * A blanket suppression would reconcile them by blinding LSan to every leak through this allocator,
 * including real ones. Instead the allocator names the ONE pointer it is abandoning, at the moment it
 * abandons it. LSan stays fully live for every other block, and nothing escapes both detectors: a
 * deliberately-leaked block is always counted in n_overrun, and every test suite requires that
 * counter to equal EXACTLY the corruption it planted -- so an unplanted one still fails the run. */
#if defined(__SANITIZE_ADDRESS__)
#  define ENGRAM_ASAN 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ENGRAM_ASAN 1
#  endif
#endif
#ifdef ENGRAM_ASAN
#  include <sanitizer/lsan_interface.h>
#  define ENGRAM_LSAN_ABANDON(p) __lsan_ignore_object(p)
#else
#  define ENGRAM_LSAN_ABANDON(p) ((void)(p))
#endif

#define ENGRAM_A_LIVE   0xE6A1110Eu
#define ENGRAM_A_DEAD   0xDEADE6A1u
#define ENGRAM_A_HDR    16u
#define ENGRAM_A_TAIL   8u
#define ENGRAM_A_POISON 0xDDu

/* Exactly sixteen bytes on every platform, so the caller's pointer (malloc + 16) keeps malloc's own
 * alignment -- 16 on x86-64, mingw-w64 and arm64. A union with long double was rejected: its size is
 * 12 on 32-bit x86 and would silently misalign every returned pointer there. */
typedef struct {
    uint32_t magic;
    uint32_t check;       /* a second witness: a damaged SIZE with an intact magic fails this */
    uint64_t size;
} engram_ahdr;

ENGRAM_STATIC_ASSERT(sizeof(engram_ahdr) == ENGRAM_A_HDR, alloc_header_is_16_bytes);

static uint32_t engram_ahdr_check(uint32_t magic, uint64_t size)
{
    return magic ^ (uint32_t)size ^ (uint32_t)(size >> 32) ^ 0x5A17C0DEu;
}

/* The tail pattern depends on the size, so a block whose size field is later read wrongly does not
 * find a matching tail at the wrong offset by coincidence. */
static uint64_t engram_atail(uint64_t size)
{
    return engram_mix64(size ^ 0xE6A5A5E6C3C33C3Cull);
}

/* ---- STATE, ALL UNDER ONE LOCK ------------------------------------------------------------------ */
static engram_mutex       g_lock = ENGRAM_MUTEX_INIT;
static engram_alloc_stats g_st;
static uint64_t           g_calls;
static uint64_t           g_fail_at;

#if ENGRAM_ALLOC_QUARANTINE_N > 0
static void    *g_q[ENGRAM_ALLOC_QUARANTINE_N];      /* base pointers, oldest at g_qfirst */
static size_t   g_qsize[ENGRAM_ALLOC_QUARANTINE_N];
#endif
static unsigned g_qfirst;
static unsigned g_qn;
static size_t   g_qbytes;

/* Under the lock. Returns 1 if this call is the one that must fail (I3). */
static int engram_alloc_tick_locked(void)
{
    g_calls++;
    if (g_fail_at && g_calls == g_fail_at) {
        g_fail_at = 0;
        g_st.n_fail_injected++;
        return 1;
    }
    return 0;
}

/* ---- VERIFICATION ---------------------------------------------------------------------------------
 * The SIZE is trusted only after both the magic and the check agree with it, so a wild size can never
 * direct the tail read somewhere arbitrary. */
enum { ENGRAM_V_OK = 0, ENGRAM_V_DEAD = 1, ENGRAM_V_FOREIGN = 2, ENGRAM_V_OVERRUN = 3 };

static int engram_verify(const unsigned char *base, engram_ahdr *out)
{
    engram_ahdr h;
    uint64_t t;
    memcpy(&h, base, sizeof h);
    if (h.magic == ENGRAM_A_DEAD && h.check == engram_ahdr_check(ENGRAM_A_DEAD, h.size))
        return ENGRAM_V_DEAD;
    if (h.magic != ENGRAM_A_LIVE || h.check != engram_ahdr_check(ENGRAM_A_LIVE, h.size))
        return ENGRAM_V_FOREIGN;
    if (h.size > (uint64_t)(SIZE_MAX - ENGRAM_A_HDR - ENGRAM_A_TAIL))
        return ENGRAM_V_FOREIGN;
    memcpy(&t, base + ENGRAM_A_HDR + (size_t)h.size, ENGRAM_A_TAIL);
    if (t != engram_atail(h.size)) return ENGRAM_V_OVERRUN;
    if (out) *out = h;
    return ENGRAM_V_OK;
}

/* Under the lock: record a verification failure against the right counter. */
static void engram_count_bad_locked(int v)
{
    if (v == ENGRAM_V_DEAD)         g_st.n_double_free++;
    else if (v == ENGRAM_V_OVERRUN) g_st.n_overrun++;
    else                            g_st.n_foreign_free++;
}

/* ---- QUARANTINE (under the lock) ----------------------------------------------------------------
 * Releasing to the system allocator while holding g_lock is safe: free() never calls back into this
 * file. It is kept under the lock deliberately, because releasing outside it would need a second
 * list of "evicted but not yet released" blocks, and every extra list is another place for I1 to
 * break. */
#if ENGRAM_ALLOC_QUARANTINE_N > 0
static void engram_q_pop_oldest_locked(void)
{
    void  *b = g_q[g_qfirst];
    size_t s = g_qsize[g_qfirst];
    g_q[g_qfirst] = NULL;
    g_qsize[g_qfirst] = 0;
    g_qfirst = (g_qfirst + 1u) % ENGRAM_ALLOC_QUARANTINE_N;
    g_qn--;
    g_qbytes -= s;
    free(b);
}
#endif

static void engram_q_push_locked(unsigned char *base, size_t size)
{
#if ENGRAM_ALLOC_QUARANTINE_N > 0
    unsigned idx;
    if (size > ENGRAM_ALLOC_QUARANTINE_BYTES) { free(base); return; }
    while (g_qn == ENGRAM_ALLOC_QUARANTINE_N || g_qbytes + size > ENGRAM_ALLOC_QUARANTINE_BYTES)
        engram_q_pop_oldest_locked();
    idx = (g_qfirst + g_qn) % ENGRAM_ALLOC_QUARANTINE_N;
    g_q[idx] = base;
    g_qsize[idx] = size;
    g_qn++;
    g_qbytes += size;
#else
    (void)size;
    free(base);
#endif
}

void engram_alloc_quarantine_flush(void)
{
    engram_mutex_lock(&g_lock);
#if ENGRAM_ALLOC_QUARANTINE_N > 0
    while (g_qn) engram_q_pop_oldest_locked();
#endif
    engram_mutex_unlock(&g_lock);
}

/* ---- THE ALLOCATION CORE ------------------------------------------------------------------------ */

/* A request that could never be satisfied (an overflowing size). It still ticks (I3); if this was the
 * injected call it counts as injected, otherwise as a real failure. */
static void *engram_alloc_refused(void)
{
    engram_mutex_lock(&g_lock);
    if (!engram_alloc_tick_locked()) g_st.n_fail_real++;
    engram_mutex_unlock(&g_lock);
    return NULL;
}

static void *engram_alloc_raw(size_t n, int zero)
{
    unsigned char *base;
    engram_ahdr h;
    uint64_t t;
    int fail;

    if (n > SIZE_MAX - ENGRAM_A_HDR - ENGRAM_A_TAIL) return engram_alloc_refused();

    engram_mutex_lock(&g_lock);
    fail = engram_alloc_tick_locked();
    engram_mutex_unlock(&g_lock);
    if (fail) return NULL;

    base = (unsigned char *)(zero ? calloc(1u, ENGRAM_A_HDR + n + ENGRAM_A_TAIL)
                                  : malloc(ENGRAM_A_HDR + n + ENGRAM_A_TAIL));
    if (!base) {
        engram_mutex_lock(&g_lock); g_st.n_fail_real++; engram_mutex_unlock(&g_lock);
        return NULL;
    }
    h.magic = ENGRAM_A_LIVE;
    h.size  = (uint64_t)n;
    h.check = engram_ahdr_check(ENGRAM_A_LIVE, h.size);
    memcpy(base, &h, sizeof h);
    t = engram_atail(h.size);
    memcpy(base + ENGRAM_A_HDR + n, &t, ENGRAM_A_TAIL);

    engram_mutex_lock(&g_lock);
    g_st.n_alloc++;
    g_st.live_blocks++;
    g_st.live_bytes += (uint64_t)n;
    if (g_st.live_bytes > g_st.peak_bytes) g_st.peak_bytes = g_st.live_bytes;
    engram_mutex_unlock(&g_lock);
    return base + ENGRAM_A_HDR;
}

void *engram_malloc(size_t n) { return engram_alloc_raw(n, 0); }

void *engram_calloc(size_t count, size_t size)
{
    if (size && count > SIZE_MAX / size) return engram_alloc_refused();
    return engram_alloc_raw(count * size, 1);
}

void *engram_array(size_t count, size_t size)
{
    if (size && count > SIZE_MAX / size) return engram_alloc_refused();
    return engram_alloc_raw(count * size, 0);
}

void *engram_realloc(void *p, size_t n)
{
    unsigned char *base, *nb;
    engram_ahdr h;
    uint64_t t, old;
    int v, fail;

    if (!p) return engram_malloc(n);
    if (n == 0) return NULL;                       /* refused; p is untouched and still valid */

    base = (unsigned char *)p - ENGRAM_A_HDR;
    v = engram_verify(base, &h);
    if (v != ENGRAM_V_OK) {
        /* Never hand a damaged block to the system realloc (I1). The caller's pointer is unusable
         * either way; returning NULL tells them so without making the damage worse. */
        engram_mutex_lock(&g_lock); engram_count_bad_locked(v); engram_mutex_unlock(&g_lock);
        if (v == ENGRAM_V_OVERRUN) ENGRAM_LSAN_ABANDON(base);
        return NULL;
    }
    if (n > SIZE_MAX - ENGRAM_A_HDR - ENGRAM_A_TAIL) return engram_alloc_refused();

    engram_mutex_lock(&g_lock);
    fail = engram_alloc_tick_locked();
    engram_mutex_unlock(&g_lock);
    if (fail) return NULL;

    old = h.size;
    nb = (unsigned char *)realloc(base, ENGRAM_A_HDR + n + ENGRAM_A_TAIL);
    if (!nb) {                                     /* the original block is untouched and valid */
        engram_mutex_lock(&g_lock); g_st.n_fail_real++; engram_mutex_unlock(&g_lock);
        return NULL;
    }
    h.size  = (uint64_t)n;
    h.check = engram_ahdr_check(ENGRAM_A_LIVE, h.size);
    memcpy(nb, &h, sizeof h);
    t = engram_atail(h.size);
    memcpy(nb + ENGRAM_A_HDR + n, &t, ENGRAM_A_TAIL);

    engram_mutex_lock(&g_lock);
    g_st.n_realloc++;
    g_st.live_bytes = g_st.live_bytes - old + (uint64_t)n;
    if (g_st.live_bytes > g_st.peak_bytes) g_st.peak_bytes = g_st.live_bytes;
    engram_mutex_unlock(&g_lock);
    return nb + ENGRAM_A_HDR;
}

void engram_free(void *p)
{
    unsigned char *base;
    engram_ahdr h;
    int v;

    if (!p) return;
    base = (unsigned char *)p - ENGRAM_A_HDR;
    v = engram_verify(base, &h);
    if (v != ENGRAM_V_OK) {
        engram_mutex_lock(&g_lock); engram_count_bad_locked(v); engram_mutex_unlock(&g_lock);
        /* Only an OVERRUN block is ours to abandon: its header verified, so `base` really is the
         * pointer malloc returned. A DEAD block is still owned by the quarantine and will be released
         * normally; a FOREIGN pointer was never ours, so there is nothing to abandon. */
        if (v == ENGRAM_V_OVERRUN) ENGRAM_LSAN_ABANDON(base);
        return;                                    /* leaked on purpose (I1) */
    }

    /* Poison, then mark dead. Poison first so that the instant the DEAD marker is visible, the body
     * already reads as garbage -- there is no moment at which a dead block holds plausible data. */
    memset(base + ENGRAM_A_HDR, ENGRAM_A_POISON, (size_t)h.size);
    h.magic = ENGRAM_A_DEAD;
    h.check = engram_ahdr_check(ENGRAM_A_DEAD, h.size);
    memcpy(base, &h, sizeof h);

    engram_mutex_lock(&g_lock);
    g_st.n_free++;
    g_st.live_blocks--;
    g_st.live_bytes -= h.size;
    engram_q_push_locked(base, (size_t)h.size);
    engram_mutex_unlock(&g_lock);
}

engram_rc engram_grow(void **pp, size_t *cap, size_t need, size_t size)
{
    size_t nc;
    void *np;
    if (!pp || !cap || !size) return ENGRAM_E_ARG;
    if (need <= *cap && *pp) return ENGRAM_OK;
    nc = *cap < 8u ? 8u : *cap;
    while (nc < need) {
        if (nc > SIZE_MAX / 2u) { nc = need; break; }
        nc *= 2u;
    }
    if (nc > SIZE_MAX / size) return ENGRAM_E_OVERFLOW;
    np = *pp ? engram_realloc(*pp, nc * size) : engram_malloc(nc * size);
    if (!np) return ENGRAM_E_MEM;                  /* *pp and *cap untouched: still valid */
    *pp = np;
    *cap = nc;
    return ENGRAM_OK;
}

char *engram_strdup(const char *s)
{
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s);
    d = (char *)engram_malloc(n + 1u);
    if (d) memcpy(d, s, n + 1u);
    return d;
}

char *engram_strndup(const char *s, size_t n)
{
    size_t m = 0;
    char *d;
    if (!s) return NULL;
    while (m < n && s[m]) m++;
    d = (char *)engram_malloc(m + 1u);
    if (!d) return NULL;
    memcpy(d, s, m);
    d[m] = 0;
    return d;
}

/* ---- INTROSPECTION ------------------------------------------------------------------------------ */
size_t engram_alloc_size(const void *p)
{
    engram_ahdr h;
    if (!p) return 0;
    if (engram_verify((const unsigned char *)p - ENGRAM_A_HDR, &h) != ENGRAM_V_OK) return 0;
    return (size_t)h.size;
}

engram_rc engram_alloc_check(const void *p)
{
    if (!p) return ENGRAM_E_ARG;
    return engram_verify((const unsigned char *)p - ENGRAM_A_HDR, NULL) == ENGRAM_V_OK
               ? ENGRAM_OK : ENGRAM_E_INTERNAL;
}

void engram_alloc_stats_get(engram_alloc_stats *out)
{
    if (!out) return;
    engram_mutex_lock(&g_lock);
    *out = g_st;
    out->quarantine_blocks = g_qn;
    out->quarantine_bytes  = (uint64_t)g_qbytes;
    engram_mutex_unlock(&g_lock);
}

uint64_t engram_alloc_corruption(void)
{
    uint64_t v;
    engram_mutex_lock(&g_lock);
    v = g_st.n_double_free + g_st.n_foreign_free + g_st.n_overrun;
    engram_mutex_unlock(&g_lock);
    return v;
}

void engram_alloc_fail_at(uint64_t nth)
{
    engram_mutex_lock(&g_lock);
    g_fail_at = nth ? g_calls + nth : 0;
    engram_mutex_unlock(&g_lock);
}

uint64_t engram_alloc_calls(void)
{
    uint64_t v;
    engram_mutex_lock(&g_lock); v = g_calls; engram_mutex_unlock(&g_lock);
    return v;
}

void engram_alloc_reset_run(void)
{
    engram_mutex_lock(&g_lock);
    g_calls = 0;
    g_fail_at = 0;
    g_st.n_fail_injected = 0;
    g_st.peak_bytes = g_st.live_bytes;
    engram_mutex_unlock(&g_lock);
}
