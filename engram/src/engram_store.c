/* ==================================================================================================
 * engram_store.c -- the episodic store. Contract in engram_store.h.
 * ==================================================================================================
 * LAYOUT. Struct of arrays over SLOTS: a slot is an episode ever added and not yet compacted away;
 * live episodes and tombstones both occupy one. Slot order is id order, always -- ids are handed out
 * increasing, slots are appended, and compaction moves slots down without reordering -- so an id is
 * found by binary search and "oldest" means "lowest slot".
 *
 * THE ADD, IN FOUR STEPS, AND WHY THE COMMIT CANNOT FAIL
 *   1. MEASURE  cut the text and sign every chunk into a stack buffer: how many chunks are storable
 *               and how many bytes they carry. Nothing allocated, nothing changed.
 *   2. PLAN     which consolidated episodes must go to make room: THE EVICTION ORDER below, as a
 *               recall threshold and a count. Not enough: ENGRAM_E_FULL, nothing changed.
 *   3. RESERVE  compact if that alone makes room (allocation-free, logically invisible -- and it keeps
 *               slot order and recall counts, so the plan still names the same episodes), else grow
 *               every array. A failed growth leaves each array's CONTENT as it was.
 *   4. WRITE, THEN COMMIT  new episodes are written into slots at and beyond n -- invisible, because
 *               everything reads only [0, n). Then the planned evictions are applied and n, the byte
 *               counts and next_id move. No step of 4 can fail.
 * ============================================================================================== */
#include "engram_store.h"
#include "engram_align.h"
#include "engram_alloc.h"
#include "engram_buf.h"
#include "engram_chunk.h"

#include <string.h>

typedef struct { double s; size_t slot; } engram_cand;     /* a stage-1 candidate */

typedef struct {                                            /* a stage-2 candidate */
    size_t   slot;
    double   contain, exact;
    uint32_t edits, aligned;
} engram_ranked;

/* How far the bag may disagree with an alignment it lets through (THE RANKING, engram_store.h). */
#define ENGRAM_ALIGN_BAG_SLACK 0.2

struct engram_store {
    engram_store_cfg cfg;
    uint64_t         geometry;
    size_t           n;                    /* slots in use: live + tombstones           */
    size_t           cap_id, cap_time, cap_src, cap_flags, cap_rec, cap_off, cap_len, cap_sig;
    uint64_t        *id, *time_ms;
    uint32_t        *source, *flags, *recalls;
    size_t          *off;
    uint16_t        *len;
    uint64_t        *sig;                  /* n * ENGRAM_SIG_WORDS                      */
    char            *text;
    size_t           text_used, text_cap;
    size_t           live, live_bytes, consolidated;
    uint64_t         next_id;
    uint64_t         adds, deletes, evictions, refusals, compactions;
    engram_encq     *q;                    /* recall scratch                            */
    engram_cand     *cand;
    engram_ranked   *rk;
    size_t           cap_cand, cap_rk;
    size_t           amax_d, amax_q;       /* alignment capacity: episode, cue (codepoints) */
    uint32_t        *acue, *atext, *awork, *ahist;
};

/* ---- THE EVICTION ORDER ---------------------------------------------------------------------
 * Among live CONSOLIDATED episodes (an unconsolidated one exists nowhere else and is never evicted):
 * the FEWEST RECALLS first, then the OLDEST. The plan is (r, take): every candidate recalled fewer
 * than r times, then the first `take` in slot order recalled exactly r times -- the shortest prefix of
 * that order that frees enough slots AND bytes. It is found without allocating: one pass fills a
 * histogram of recall counts 0..62 (63 collects the rest); only when the threshold lies in that last
 * bucket -- every cheaper candidate spent and the rest recalled 63+ times -- does a binary search over
 * the counts run. The commit re-applies the same rule; nothing between plan and commit changes it. */
typedef struct { uint32_t r; size_t take; int any; } engram_evict_plan;

#define ENGRAM_EVICT_BUCKETS 64u

static int engram_evictable(const engram_store *s, size_t slot)
{
    return (s->flags[slot] & (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) ==
           (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED);
}

/* slots and bytes of the evictable episodes recalled at most r times */
static void engram_evict_tally(const engram_store *s, uint32_t r, size_t *slots, size_t *bytes)
{
    size_t i;
    *slots = 0; *bytes = 0;
    for (i = 0; i < s->n; i++)
        if (engram_evictable(s, i) && s->recalls[i] <= r) { (*slots)++; *bytes += s->len[i]; }
}

static int engram_evict_plan_make(const engram_store *s, size_t need_slots, size_t need_bytes,
                                  engram_evict_plan *p)
{
    size_t hs[ENGRAM_EVICT_BUCKETS], hb[ENGRAM_EVICT_BUCKETS], fs = 0, fb = 0, i;
    uint32_t r = 0, b, maxr = 0;
    p->r = 0; p->take = 0; p->any = need_slots > 0u || need_bytes > 0u;
    if (!p->any) return 1;
    memset(hs, 0, sizeof hs); memset(hb, 0, sizeof hb);
    for (i = 0; i < s->n; i++)
        if (engram_evictable(s, i)) {
            uint32_t rc = s->recalls[i];
            b = rc < ENGRAM_EVICT_BUCKETS - 1u ? rc : ENGRAM_EVICT_BUCKETS - 1u;
            hs[b]++; hb[b] += s->len[i];
            if (rc > maxr) maxr = rc;
        }
    for (b = 0; b < ENGRAM_EVICT_BUCKETS; b++) {      /* the smallest r whose "<= r" set suffices */
        if (fs + hs[b] >= need_slots && fb + hb[b] >= need_bytes) break;
        fs += hs[b]; fb += hb[b];
    }
    if (b == ENGRAM_EVICT_BUCKETS) return 0;          /* even every candidate is not enough */
    r = b;
    if (b == ENGRAM_EVICT_BUCKETS - 1u) {             /* inside the 63+ bucket: search the counts */
        uint32_t lo = b, hi = maxr;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2u;
            size_t ts, tb;
            engram_evict_tally(s, mid, &ts, &tb);
            if (ts >= need_slots && tb >= need_bytes) hi = mid; else lo = mid + 1u;
        }
        r = lo;
        if (r > 0u) engram_evict_tally(s, r - 1u, &fs, &fb); else { fs = 0; fb = 0; }
    }
    for (i = 0; i < s->n && (fs < need_slots || fb < need_bytes); i++)   /* then the oldest at r */
        if (engram_evictable(s, i) && s->recalls[i] == r) { fs++; fb += s->len[i]; p->take++; }
    p->r = r;
    return 1;
}


void engram_store_cfg_default(engram_store_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->max_episodes   = 100000u;
    cfg->max_text_bytes = (size_t)64u << 20;
    cfg->chunk_cap      = ENGRAM_EPI_TAIL;
    cfg->chunk_overlap  = 0u;
    cfg->recall_c       = 50u;
    cfg->rank           = ENGRAM_RANK_FULL;
    engram_enc_cfg_default(&cfg->enc);
}

static engram_rc engram_store_cfg_check(const engram_store_cfg *c)
{
    engram_chunkit it;
    if (!c || !c->max_episodes || !c->max_text_bytes || !c->recall_c) return ENGRAM_E_ARG;
    if (c->rank != ENGRAM_RANK_FULL && c->rank != ENGRAM_RANK_BAG && c->rank != ENGRAM_RANK_EXACT)
        return ENGRAM_E_ARG;
    if (engram_enc_cfg_check(&c->enc) != ENGRAM_OK || c->enc.tf != ENGRAM_TF_SQRT) return ENGRAM_E_ARG;
    return engram_chunkit_init(&it, "", 0u, c->chunk_cap, c->chunk_overlap);
}

engram_rc engram_store_open(engram_store **out, const engram_store_cfg *cfg)
{
    engram_store *s;
    engram_store_cfg d;
    engram_rc rc;
    if (!out) return ENGRAM_E_ARG;
    *out = NULL;
    if (!cfg) { engram_store_cfg_default(&d); cfg = &d; }
    rc = engram_store_cfg_check(cfg);
    if (rc != ENGRAM_OK) return rc;
    s = (engram_store *)engram_malloc(sizeof *s);
    if (!s) return ENGRAM_E_MEM;
    memset(s, 0, sizeof *s);
    s->cfg = *cfg;
    s->geometry = engram_enc_geometry(&cfg->enc, ENGRAM_D);
    s->next_id = 1u;
    /* Alignment buffers, sized by proof rather than by hope: an episode of b bytes normalises to at
     * most ENGRAM_FOLD_OUT_MAX * b codepoints (test_align), and a cue longer than 3/2 of that can
     * never be aligned -- edits >= |q| - |d| > |q| / 3 -- so it is not aligned at all, which is the
     * same answer without the work. */
    s->amax_d = ENGRAM_FOLD_OUT_MAX * cfg->chunk_cap;
    s->amax_q = s->amax_d * 3u / 2u;
    s->q = (engram_encq *)engram_malloc(sizeof *s->q);
    s->acue = (uint32_t *)engram_array(s->amax_q, sizeof *s->acue);
    s->atext = (uint32_t *)engram_array(s->amax_d, sizeof *s->atext);
    s->awork = (uint32_t *)engram_array(3u * (s->amax_d + 1u), sizeof *s->awork);
    s->ahist = (uint32_t *)engram_array(s->amax_q + 1u, sizeof *s->ahist);
    if (!s->q || !s->acue || !s->atext || !s->awork || !s->ahist) { engram_store_close(s); return ENGRAM_E_MEM; }
    *out = s;
    return ENGRAM_OK;
}

void engram_store_close(engram_store *s)
{
    if (!s) return;
    engram_free(s->id); engram_free(s->time_ms); engram_free(s->source); engram_free(s->flags);
    engram_free(s->recalls); engram_free(s->off); engram_free(s->len); engram_free(s->sig);
    engram_free(s->text); engram_free(s->q); engram_free(s->cand); engram_free(s->rk);
    engram_free(s->acue); engram_free(s->atext); engram_free(s->awork); engram_free(s->ahist);
    engram_free(s);
}

/* ---- lookup ---------------------------------------------------------------------------------- */
static int engram_store_slot(const engram_store *s, uint64_t id, size_t *slot)
{
    size_t lo = 0, hi = s->n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2u;
        if (s->id[mid] < id) lo = mid + 1u; else hi = mid;
    }
    if (lo < s->n && s->id[lo] == id && (s->flags[lo] & ENGRAM_EPI_LIVE)) { *slot = lo; return 1; }
    return 0;
}

/* ---- compaction: slots move down in order, text moves down in order; nothing is allocated ---- */
void engram_store_compact(engram_store *s)
{
    size_t r, w = 0, tw = 0;
    if (!s || s->n == s->live) return;
    for (r = 0; r < s->n; r++) {
        if (!(s->flags[r] & ENGRAM_EPI_LIVE)) continue;
        if (s->off[r] != tw) memmove(s->text + tw, s->text + s->off[r], s->len[r]);
        if (w != r) {
            s->id[w] = s->id[r]; s->time_ms[w] = s->time_ms[r]; s->source[w] = s->source[r];
            s->flags[w] = s->flags[r]; s->recalls[w] = s->recalls[r]; s->len[w] = s->len[r];
            memcpy(s->sig + w * ENGRAM_SIG_WORDS, s->sig + r * ENGRAM_SIG_WORDS,
                   ENGRAM_SIG_WORDS * sizeof *s->sig);
        }
        s->off[w] = tw;
        tw += s->len[r];
        w++;
    }
    s->n = w;
    s->text_used = tw;
    s->compactions++;
}

/* ---- add ------------------------------------------------------------------------------------- */
static engram_rc engram_store_reserve(engram_store *s, size_t slots, size_t bytes)
{
    size_t need = s->n + slots;
    if (engram_grow((void **)&s->id, &s->cap_id, need, sizeof *s->id) != ENGRAM_OK ||
        engram_grow((void **)&s->time_ms, &s->cap_time, need, sizeof *s->time_ms) != ENGRAM_OK ||
        engram_grow((void **)&s->source, &s->cap_src, need, sizeof *s->source) != ENGRAM_OK ||
        engram_grow((void **)&s->flags, &s->cap_flags, need, sizeof *s->flags) != ENGRAM_OK ||
        engram_grow((void **)&s->recalls, &s->cap_rec, need, sizeof *s->recalls) != ENGRAM_OK ||
        engram_grow((void **)&s->off, &s->cap_off, need, sizeof *s->off) != ENGRAM_OK ||
        engram_grow((void **)&s->len, &s->cap_len, need, sizeof *s->len) != ENGRAM_OK ||
        engram_grow((void **)&s->sig, &s->cap_sig, need * ENGRAM_SIG_WORDS, sizeof *s->sig) != ENGRAM_OK ||
        engram_grow((void **)&s->text, &s->text_cap, s->text_used + bytes + 1u, 1u) != ENGRAM_OK)
        return ENGRAM_E_MEM;
    return ENGRAM_OK;
}

engram_rc engram_store_add(engram_store *s, const void *text, size_t n, uint64_t time_ms,
                           uint32_t source, uint64_t *first_id, size_t *n_added)
{
    uint64_t tmp[ENGRAM_SIG_WORDS];
    engram_chunkit it;
    size_t off, len, k = 0, bytes = 0, slot;
    engram_evict_plan plan;
    engram_rc rc, last = ENGRAM_E_SHORT;

    if (first_id) *first_id = 0;
    if (n_added) *n_added = 0;
    if (!s || !first_id || !n_added || (!text && n)) return ENGRAM_E_ARG;

    /* 1. MEASURE */
    rc = engram_chunkit_init(&it, text, n, s->cfg.chunk_cap, s->cfg.chunk_overlap);
    if (rc != ENGRAM_OK) return rc;
    while (engram_chunkit_next(&it, &off, &len)) {
        rc = engram_encode_sig(&s->cfg.enc, (const char *)text + off, len, tmp, NULL);
        if (rc == ENGRAM_OK) { k++; bytes += len; }
        else if (rc == ENGRAM_E_UTF8) return rc;
        else if (rc != ENGRAM_E_SHORT) return rc;
    }
    if (k == 0) return last;

    /* 2. PLAN: THE EVICTION ORDER */
    if (k > s->cfg.max_episodes || bytes > s->cfg.max_text_bytes) { s->refusals++; return ENGRAM_E_FULL; }
    {
        size_t need_slots = s->live + k > s->cfg.max_episodes ? s->live + k - s->cfg.max_episodes : 0u;
        size_t need_bytes = s->live_bytes + bytes > s->cfg.max_text_bytes
                                ? s->live_bytes + bytes - s->cfg.max_text_bytes : 0u;
        if (!engram_evict_plan_make(s, need_slots, need_bytes, &plan)) { s->refusals++; return ENGRAM_E_FULL; }
    }

    /* 3. RESERVE. Compaction is logically invisible, so it may run before a later failure; it keeps
     * slot order and recall counts, so the plan still names the same episodes afterwards. */
    if (s->n > s->live &&
        (s->n + k > s->cap_id || s->text_used + bytes + 1u > s->text_cap))
        engram_store_compact(s);
    rc = engram_store_reserve(s, k, bytes);
    if (rc != ENGRAM_OK) return rc;

    /* 4. WRITE into slots n .. n+k-1 (invisible), THEN COMMIT */
    {
        size_t w = s->n, tw = s->text_used;
        uint64_t nid = s->next_id;
        engram_chunkit it2;
        (void)engram_chunkit_init(&it2, text, n, s->cfg.chunk_cap, s->cfg.chunk_overlap);
        while (engram_chunkit_next(&it2, &off, &len)) {
            if (engram_encode_sig(&s->cfg.enc, (const char *)text + off, len,
                                  s->sig + w * ENGRAM_SIG_WORDS, NULL) != ENGRAM_OK) continue;
            memcpy(s->text + tw, (const char *)text + off, len);
            s->id[w] = nid++;
            s->time_ms[w] = time_ms;
            s->source[w] = source;
            s->flags[w] = ENGRAM_EPI_LIVE;
            s->recalls[w] = 0;
            s->off[w] = tw;
            s->len[w] = (uint16_t)len;
            tw += len;
            w++;
        }
        /* commit */
        for (slot = 0; plan.any && slot < s->n; slot++)
            if (engram_evictable(s, slot) &&
                (s->recalls[slot] < plan.r || (s->recalls[slot] == plan.r && plan.take > 0u))) {
                if (s->recalls[slot] == plan.r) plan.take--;
                s->flags[slot] &= ~(uint32_t)(ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED);
                memset(s->sig + slot * ENGRAM_SIG_WORDS, 0, ENGRAM_SIG_WORDS * sizeof *s->sig);
                s->live--;
                s->live_bytes -= s->len[slot];
                s->consolidated--;
                s->evictions++;
            }
        *first_id = s->next_id;
        *n_added = k;
        s->n = w;
        s->text_used = tw;
        s->live += k;
        s->live_bytes += bytes;
        s->next_id = nid;
        s->adds++;
    }
    /* A store that evicts on every add never takes the pre-add compaction, so tombstones and their
     * text would accumulate without bound. After the commit, compaction is still allocation-free and
     * logically invisible, so the bound is enforced here: tombstones never exceed a quarter of the
     * live episodes plus a small constant. */
    if (s->n - s->live > s->live / 4u + 64u) engram_store_compact(s);
    return ENGRAM_OK;
}

/* ---- the other mutations --------------------------------------------------------------------- */
engram_rc engram_store_delete(engram_store *s, uint64_t id)
{
    size_t slot;
    if (!s) return ENGRAM_E_ARG;
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    if (s->flags[slot] & ENGRAM_EPI_CONSOLIDATED) s->consolidated--;
    s->flags[slot] &= ~(uint32_t)(ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED);
    memset(s->sig + slot * ENGRAM_SIG_WORDS, 0, ENGRAM_SIG_WORDS * sizeof *s->sig);
    s->live--;
    s->live_bytes -= s->len[slot];
    s->deletes++;
    return ENGRAM_OK;
}

engram_rc engram_store_consolidated(engram_store *s, uint64_t id)
{
    size_t slot;
    if (!s) return ENGRAM_E_ARG;
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    if (!(s->flags[slot] & ENGRAM_EPI_CONSOLIDATED)) { s->flags[slot] |= ENGRAM_EPI_CONSOLIDATED; s->consolidated++; }
    return ENGRAM_OK;
}

engram_rc engram_store_get(const engram_store *s, uint64_t id, engram_episode *out)
{
    size_t slot;
    if (!s || !out) return ENGRAM_E_ARG;
    memset(out, 0, sizeof *out);
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    out->id = s->id[slot]; out->time_ms = s->time_ms[slot]; out->source = s->source[slot];
    out->flags = s->flags[slot]; out->recalls = s->recalls[slot];
    out->text = s->text + s->off[slot]; out->len = s->len[slot];
    return ENGRAM_OK;
}

/* ---- recall ----------------------------------------------------------------------------------
 * Stage 1: every live episode's signature containment. The C best, by a size-C min-heap whose
 * order is (score, then LOWER slot wins) -- a total order, so the candidate set is deterministic.
 * Stage 2: THE RANKING (engram_store.h) over those C, sorted by a total order, the k best returned. */
static int engram_cand_worse(const engram_cand *a, const engram_cand *b)
{
    return a->s < b->s || (a->s == b->s && a->slot > b->slot);
}

static void engram_heap_sift(engram_cand *h, size_t n, size_t i)
{
    for (;;) {
        size_t l = 2u * i + 1u, r = l + 1u, m = i;
        engram_cand t;
        if (l < n && engram_cand_worse(&h[l], &h[m])) m = l;
        if (r < n && engram_cand_worse(&h[r], &h[m])) m = r;
        if (m == i) return;
        t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* 1 if a ranks before b. A total order: no two candidates share a slot. */
static int engram_ranked_before(const engram_ranked *a, const engram_ranked *b, engram_rank mode)
{
    if (mode == ENGRAM_RANK_FULL) {
        if (a->aligned != b->aligned) return a->aligned > b->aligned;
        if (a->aligned && a->edits != b->edits) return a->edits < b->edits;
    }
    if (mode != ENGRAM_RANK_EXACT && a->contain != b->contain) return a->contain > b->contain;
    if (a->exact != b->exact) return a->exact > b->exact;
    return a->slot < b->slot;
}

/* Heapsort into best-first order: the root of the heap is the candidate that ranks LAST. */
static void engram_ranked_sift(engram_ranked *h, size_t n, size_t i, engram_rank mode)
{
    for (;;) {
        size_t l = 2u * i + 1u, r = l + 1u, m = i;
        engram_ranked t;
        if (l < n && engram_ranked_before(&h[m], &h[l], mode)) m = l;
        if (r < n && engram_ranked_before(&h[m], &h[r], mode)) m = r;
        if (m == i) return;
        t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

static void engram_ranked_sort(engram_ranked *h, size_t n, engram_rank mode)
{
    size_t i;
    if (n < 2u) return;
    for (i = n / 2u; i-- > 0u; ) engram_ranked_sift(h, n, i, mode);
    for (i = n - 1u; i > 0u; i--) {
        engram_ranked t = h[0]; h[0] = h[i]; h[i] = t;
        engram_ranked_sift(h, i, 0u, mode);
    }
}

/* Mark the significant alignments among rk[0..m): see THE RANKING. edits are all <= qn <= amax_q. */
static void engram_mark_aligned(engram_store *s, engram_ranked *rk, size_t m, size_t qn)
{
    size_t i, lo = (m - 1u) / 2u, hi = m / 2u, seen = 0;
    uint32_t e, mlo = 0, mhi = 0;
    uint64_t med2;
    double best = 0.0;
    int got_lo = 0;
    memset(s->ahist, 0, (qn + 1u) * sizeof *s->ahist);
    for (i = 0; i < m; i++) {
        s->ahist[rk[i].edits]++;
        if (rk[i].contain > best) best = rk[i].contain;
    }
    for (e = 0; e <= (uint32_t)qn; e++) {             /* the two middle order statistics */
        seen += s->ahist[e];
        if (!got_lo && seen > lo) { mlo = e; got_lo = 1; }
        if (seen > hi) { mhi = e; break; }
    }
    med2 = (uint64_t)mlo + mhi;                       /* twice the median */
    for (i = 0; i < m; i++) {
        uint64_t ed = rk[i].edits;
        rk[i].aligned = (6u * ed <= med2 && 3u * ed <= (uint64_t)qn &&
                         rk[i].contain >= best - ENGRAM_ALIGN_BAG_SLACK) ? 1u : 0u;
    }
}

engram_rc engram_store_recall(engram_store *s, const void *query, size_t n, engram_hit *hits,
                              size_t k, size_t *n_hits)
{
    engram_cand *heap;
    engram_ranked *rk;
    size_t C, hn = 0, slot, i, qn = 0;
    int align = 0;
    engram_rc rc;
    if (n_hits) *n_hits = 0;
    if (!s || !hits || !n_hits || !k || (!query && n)) return ENGRAM_E_ARG;
    rc = engram_encq_build(s->q, &s->cfg.enc, query, n);
    if (rc != ENGRAM_OK) return rc;
    if (s->live == 0) return ENGRAM_OK;
    C = s->cfg.recall_c > k ? s->cfg.recall_c : k;
    if (engram_grow((void **)&s->cand, &s->cap_cand, C, sizeof *s->cand) != ENGRAM_OK ||
        engram_grow((void **)&s->rk, &s->cap_rk, C, sizeof *s->rk) != ENGRAM_OK) return ENGRAM_E_MEM;
    if (s->cfg.rank == ENGRAM_RANK_FULL) {
        rc = engram_align_norm(query, n, s->cfg.enc.fold, s->cfg.enc.utf8, s->acue, s->amax_q, &qn);
        if (rc == ENGRAM_OK) align = qn > 0u;
        else if (rc != ENGRAM_E_FULL) return rc;      /* E_FULL: too long to align -- see open */
    }
    heap = s->cand;
    rk = s->rk;
    for (slot = 0; slot < s->n; slot++) {
        engram_cand c;
        if (!(s->flags[slot] & ENGRAM_EPI_LIVE)) continue;
        (void)engram_encq_sig_score(s->q, s->sig + slot * ENGRAM_SIG_WORDS, &c.s);
        c.slot = slot;
        if (hn < C) {
            size_t x = hn++;
            heap[x] = c;
            while (x > 0u && engram_cand_worse(&heap[x], &heap[(x - 1u) / 2u])) {
                engram_cand t = heap[x]; heap[x] = heap[(x - 1u) / 2u]; heap[(x - 1u) / 2u] = t;
                x = (x - 1u) / 2u;
            }
        } else if (engram_cand_worse(&heap[0], &c)) {
            heap[0] = c;
            engram_heap_sift(heap, hn, 0u);
        }
    }
    /* stage 2. A stored episode always encodes (it was signed on the way in), so the scores and the
     * normalisation cannot fail; if one ever did, the candidate would rank with zeros and edits = |q|,
     * the worst values, rather than take the recall down. */
    for (i = 0; i < hn; i++) {
        const char *t = s->text + s->off[heap[i].slot];
        size_t tl = s->len[heap[i].slot], dn = 0;
        rk[i].slot = heap[i].slot;
        (void)engram_encq_scores(s->q, t, tl, &rk[i].exact, &rk[i].contain);
        rk[i].edits = ENGRAM_EDITS_NONE;
        rk[i].aligned = 0u;
        if (align)
            rk[i].edits = engram_align_norm(t, tl, s->cfg.enc.fold, s->cfg.enc.utf8, s->atext, s->amax_d, &dn)
                              == ENGRAM_OK ? engram_align_dist(s->acue, qn, s->atext, dn, s->awork) : (uint32_t)qn;
    }
    if (align) engram_mark_aligned(s, rk, hn, qn);
    engram_ranked_sort(rk, hn, s->cfg.rank);
    for (i = 0; i < hn && i < k; i++) {
        hits[i].id = s->id[rk[i].slot];
        hits[i].contain = rk[i].contain;
        hits[i].exact = rk[i].exact;
        hits[i].edits = rk[i].edits;
        hits[i].aligned = rk[i].aligned;
    }
    *n_hits = i;
    if (i > 0u) s->recalls[rk[0].slot]++;
    return ENGRAM_OK;
}

/* ---- stats and fingerprint ------------------------------------------------------------------- */
void engram_store_stats_get(const engram_store *s, engram_store_stats *st)
{
    if (!st) return;
    memset(st, 0, sizeof *st);
    if (!s) return;
    st->live = s->live; st->tombstones = s->n - s->live; st->capacity = s->cap_id;
    st->text_bytes = s->live_bytes; st->text_capacity = s->text_cap; st->consolidated = s->consolidated;
    st->next_id = s->next_id; st->adds = s->adds; st->deletes = s->deletes; st->evictions = s->evictions;
    st->refusals = s->refusals; st->compactions = s->compactions;
    st->bytes_resident = sizeof *s + s->cap_id * 8u + s->cap_time * 8u + s->cap_src * 4u + s->cap_flags * 4u +
                         s->cap_rec * 4u + s->cap_off * sizeof(size_t) + s->cap_len * 2u + s->cap_sig * 8u +
                         s->text_cap + sizeof *s->q + s->cap_cand * sizeof *s->cand +
                         s->cap_rk * sizeof *s->rk + (s->amax_q + s->amax_d + 3u * (s->amax_d + 1u) +
                         s->amax_q + 1u) * sizeof(uint32_t);
}

uint64_t engram_store_fingerprint(const engram_store *s)
{
    uint64_t h;
    size_t slot;
    if (!s) return 0;
    h = engram_mix2(0x53544F5245ull, s->geometry);
    h = engram_mix2(h, (uint64_t)s->cfg.max_episodes);
    h = engram_mix2(h, (uint64_t)s->cfg.max_text_bytes);
    h = engram_mix2(h, ((uint64_t)s->cfg.chunk_cap << 32) | (uint64_t)s->cfg.chunk_overlap);
    h = engram_mix2(h, ((uint64_t)s->cfg.recall_c << 8) | (uint64_t)s->cfg.rank);
    h = engram_mix2(h, s->next_id);
    h = engram_mix2(h, (uint64_t)s->live);
    for (slot = 0; slot < s->n; slot++) {
        if (!(s->flags[slot] & ENGRAM_EPI_LIVE)) continue;
        h = engram_mix2(h, s->id[slot]);
        h = engram_mix2(h, s->time_ms[slot]);
        h = engram_mix2(h, ((uint64_t)s->source[slot] << 32) | s->flags[slot]);
        h = engram_mix2(h, ((uint64_t)s->recalls[slot] << 32) | s->len[slot]);
        h = engram_mix2(h, engram_hash_bytes(s->text + s->off[slot], s->len[slot], 0u));
        h = engram_mix2(h, engram_hash_bytes(s->sig + slot * ENGRAM_SIG_WORDS,
                                             ENGRAM_SIG_WORDS * sizeof *s->sig, 1u));
    }
    return h;
}

/* ---- persistence ------------------------------------------------------------------------------
 * PAYLOAD, version 1, little-endian (engram_buf.h):
 *   u32 version
 *   u64 max_episodes, u64 max_text_bytes, u32 chunk_cap, u32 chunk_overlap, u32 recall_c, u32 rank
 *   f32 w_char3, w_char4, w_word1, w_word2, w_ideo1, w_ideo2; u32 ordered_pairs, fold, utf8; u64 seed;
 *   u32 tf
 *   u64 next_id, adds, deletes, evictions, refusals, compactions
 *   u64 live
 *   live x { u64 id, u64 time_ms, u32 source, u32 flags, u32 recalls, u32 len, len bytes of text }
 * The load runs in two passes over the bytes: the first proves the whole payload and totals it without
 * allocating a thing; only then is the store opened at its exact size and the second pass fills it. */
#define ENGRAM_STORE_FILE_VERSION 1u
#define ENGRAM_STORE_REC_MIN      (8u + 8u + 4u + 4u + 4u + 4u + 1u)   /* the smallest episode record */

static void engram_store_put_cfg(engram_wbuf *w, const engram_store_cfg *c)
{
    engram_wbuf_u64(w, (uint64_t)c->max_episodes);
    engram_wbuf_u64(w, (uint64_t)c->max_text_bytes);
    engram_wbuf_u32(w, (uint32_t)c->chunk_cap);
    engram_wbuf_u32(w, (uint32_t)c->chunk_overlap);
    engram_wbuf_u32(w, (uint32_t)c->recall_c);
    engram_wbuf_u32(w, (uint32_t)c->rank);
    engram_wbuf_f32(w, c->enc.w_char3);
    engram_wbuf_f32(w, c->enc.w_char4);
    engram_wbuf_f32(w, c->enc.w_word1);
    engram_wbuf_f32(w, c->enc.w_word2);
    engram_wbuf_f32(w, c->enc.w_ideo1);
    engram_wbuf_f32(w, c->enc.w_ideo2);
    engram_wbuf_u32(w, (uint32_t)c->enc.ordered_pairs);
    engram_wbuf_u32(w, (uint32_t)c->enc.fold);
    engram_wbuf_u32(w, (uint32_t)c->enc.utf8);
    engram_wbuf_u64(w, c->enc.seed);
    engram_wbuf_u32(w, (uint32_t)c->enc.tf);
}

/* a u64 from the file as a size_t, refusing what does not fit */
static size_t engram_store_rsize(engram_rbuf *r)
{
    uint64_t v = engram_rbuf_u64(r);
    size_t s = (size_t)v;
    if ((uint64_t)s != v) { engram_rbuf_fail(r, ENGRAM_E_FORMAT); return 0; }
    return s;
}

static void engram_store_get_cfg(engram_rbuf *r, engram_store_cfg *c)
{
    uint32_t v;
    memset(c, 0, sizeof *c);
    c->max_episodes = engram_store_rsize(r);
    c->max_text_bytes = engram_store_rsize(r);
    c->chunk_cap = engram_rbuf_u32(r);
    c->chunk_overlap = engram_rbuf_u32(r);
    c->recall_c = engram_rbuf_u32(r);
    v = engram_rbuf_u32(r);
    c->rank = v == 0u ? ENGRAM_RANK_FULL : v == 1u ? ENGRAM_RANK_BAG : ENGRAM_RANK_EXACT;
    if (v > 2u) engram_rbuf_fail(r, ENGRAM_E_FORMAT);
    c->enc.w_char3 = engram_rbuf_f32_finite(r);
    c->enc.w_char4 = engram_rbuf_f32_finite(r);
    c->enc.w_word1 = engram_rbuf_f32_finite(r);
    c->enc.w_word2 = engram_rbuf_f32_finite(r);
    c->enc.w_ideo1 = engram_rbuf_f32_finite(r);
    c->enc.w_ideo2 = engram_rbuf_f32_finite(r);
    v = engram_rbuf_u32(r);
    if (v > 1u) engram_rbuf_fail(r, ENGRAM_E_FORMAT);
    c->enc.ordered_pairs = (int)v;
    v = engram_rbuf_u32(r);
    c->enc.fold = v == (uint32_t)ENGRAM_FOLD_NONE ? ENGRAM_FOLD_NONE
                : v == (uint32_t)ENGRAM_FOLD_CASE ? ENGRAM_FOLD_CASE : ENGRAM_FOLD_COMPAT;
    if (v != (uint32_t)c->enc.fold) engram_rbuf_fail(r, ENGRAM_E_FORMAT);
    v = engram_rbuf_u32(r);
    c->enc.utf8 = v == (uint32_t)ENGRAM_UTF8_STRICT ? ENGRAM_UTF8_STRICT : ENGRAM_UTF8_REPLACE;
    if (v != (uint32_t)c->enc.utf8) engram_rbuf_fail(r, ENGRAM_E_FORMAT);
    c->enc.seed = engram_rbuf_u64(r);
    v = engram_rbuf_u32(r);
    c->enc.tf = v == (uint32_t)ENGRAM_TF_LINEAR ? ENGRAM_TF_LINEAR : v == (uint32_t)ENGRAM_TF_SQRT ? ENGRAM_TF_SQRT
              : v == (uint32_t)ENGRAM_TF_QUARTER ? ENGRAM_TF_QUARTER : ENGRAM_TF_SIGN;
    if (v != (uint32_t)c->enc.tf) engram_rbuf_fail(r, ENGRAM_E_FORMAT);
}

engram_rc engram_store_serialize(const engram_store *s, uint8_t **out, size_t *n)
{
    engram_wbuf w;
    size_t slot;
    if (out) *out = NULL;
    if (n) *n = 0;
    if (!s || !out || !n) return ENGRAM_E_ARG;
    engram_wbuf_init(&w);
    engram_wbuf_reserve(&w, 160u + s->live * ENGRAM_STORE_REC_MIN + s->live_bytes);
    engram_wbuf_u32(&w, ENGRAM_STORE_FILE_VERSION);
    engram_store_put_cfg(&w, &s->cfg);
    engram_wbuf_u64(&w, s->next_id);
    engram_wbuf_u64(&w, s->adds);
    engram_wbuf_u64(&w, s->deletes);
    engram_wbuf_u64(&w, s->evictions);
    engram_wbuf_u64(&w, s->refusals);
    engram_wbuf_u64(&w, s->compactions);
    engram_wbuf_u64(&w, (uint64_t)s->live);
    for (slot = 0; slot < s->n; slot++) {
        if (!(s->flags[slot] & ENGRAM_EPI_LIVE)) continue;
        engram_wbuf_u64(&w, s->id[slot]);
        engram_wbuf_u64(&w, s->time_ms[slot]);
        engram_wbuf_u32(&w, s->source[slot]);
        engram_wbuf_u32(&w, s->flags[slot]);
        engram_wbuf_u32(&w, s->recalls[slot]);
        engram_wbuf_u32(&w, (uint32_t)s->len[slot]);
        engram_wbuf_bytes(&w, s->text + s->off[slot], s->len[slot]);
    }
    return engram_wbuf_finish(&w, out, n);
}

engram_rc engram_store_deserialize(engram_store **out, const void *p, size_t n)
{
    engram_rbuf r;
    engram_store_cfg cfg;
    engram_store *s = NULL;
    uint64_t next_id, adds, deletes, evictions, refusals, compactions, prev = 0;
    size_t live, i, bytes = 0, cons = 0, body;
    uint32_t ver;
    engram_rc rc;
    if (out) *out = NULL;
    if (!out || (!p && n)) return ENGRAM_E_ARG;
    engram_rbuf_init(&r, p, n);
    ver = engram_rbuf_u32(&r);
    if (engram_rbuf_status(&r) == ENGRAM_OK && ver != ENGRAM_STORE_FILE_VERSION)
        engram_rbuf_fail(&r, ver == 0u ? ENGRAM_E_FORMAT : ENGRAM_E_VERSION);
    engram_store_get_cfg(&r, &cfg);
    next_id = engram_rbuf_u64(&r);
    adds = engram_rbuf_u64(&r);
    deletes = engram_rbuf_u64(&r);
    evictions = engram_rbuf_u64(&r);
    refusals = engram_rbuf_u64(&r);
    compactions = engram_rbuf_u64(&r);
    live = engram_store_rsize(&r);
    if ((rc = engram_rbuf_status(&r)) != ENGRAM_OK) return rc;
    if (engram_store_cfg_check(&cfg) != ENGRAM_OK) return ENGRAM_E_FORMAT;
    if (next_id == 0u || live > cfg.max_episodes || live > engram_rbuf_left(&r) / ENGRAM_STORE_REC_MIN)
        return ENGRAM_E_FORMAT;

    /* pass 1: prove every record, allocate nothing */
    body = engram_rbuf_pos(&r);
    for (i = 0; i < live; i++) {
        uint64_t id = engram_rbuf_u64(&r);
        uint32_t flags, len;
        (void)engram_rbuf_u64(&r);                     /* time: any value */
        (void)engram_rbuf_u32(&r);                     /* source: any value */
        flags = engram_rbuf_u32(&r);
        (void)engram_rbuf_u32(&r);                     /* recalls: any value */
        len = engram_rbuf_u32(&r);
        if (engram_rbuf_status(&r) != ENGRAM_OK) break;
        if (id == 0u || id <= prev || id >= next_id ||
            (flags != ENGRAM_EPI_LIVE && flags != (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) ||
            len == 0u || len > cfg.chunk_cap || len > cfg.max_text_bytes - bytes) {
            engram_rbuf_fail(&r, ENGRAM_E_FORMAT);
            break;
        }
        if (!engram_rbuf_skip(&r, len)) break;
        prev = id;
        bytes += len;
        if (flags & ENGRAM_EPI_CONSOLIDATED) cons++;
    }
    if ((rc = engram_rbuf_end(&r)) != ENGRAM_OK) return rc;

    /* pass 2: open at the exact size and fill; a text that does not encode is refused */
    rc = engram_store_open(&s, &cfg);
    if (rc != ENGRAM_OK) return rc == ENGRAM_E_MEM ? rc : ENGRAM_E_FORMAT;
    if (live > 0u && engram_store_reserve(s, live, bytes) != ENGRAM_OK) { engram_store_close(s); return ENGRAM_E_MEM; }
    engram_rbuf_init(&r, p, n);
    (void)engram_rbuf_skip(&r, body);
    for (i = 0; i < live; i++) {
        const uint8_t *t;
        uint32_t len;
        s->id[i] = engram_rbuf_u64(&r);
        s->time_ms[i] = engram_rbuf_u64(&r);
        s->source[i] = engram_rbuf_u32(&r);
        s->flags[i] = engram_rbuf_u32(&r);
        s->recalls[i] = engram_rbuf_u32(&r);
        len = engram_rbuf_u32(&r);
        t = engram_rbuf_view(&r, len);
        if (!t) { engram_store_close(s); return ENGRAM_E_INTERNAL; }        /* pass 1 proved it */
        rc = engram_encode_sig(&cfg.enc, t, len, s->sig + i * ENGRAM_SIG_WORDS, NULL);
        if (rc != ENGRAM_OK) { engram_store_close(s); return ENGRAM_E_FORMAT; }
        memcpy(s->text + s->text_used, t, len);
        s->off[i] = s->text_used;
        s->len[i] = (uint16_t)len;
        s->text_used += len;
    }
    s->n = live;
    s->live = live;
    s->live_bytes = bytes;
    s->consolidated = cons;
    s->next_id = next_id;
    s->adds = adds; s->deletes = deletes; s->evictions = evictions;
    s->refusals = refusals; s->compactions = compactions;
    *out = s;
    return ENGRAM_OK;
}

engram_rc engram_store_save(const engram_store *s, const char *path, const engram_key *key)
{
    uint8_t *p = NULL;
    size_t pn = 0;
    engram_rc rc;
    if (!s || !path) return ENGRAM_E_ARG;
    rc = engram_store_serialize(s, &p, &pn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_seal_write(path, ENGRAM_KIND_STORE, key, p, pn);
    engram_wipe(p, pn);
    engram_free(p);
    return rc;
}

engram_rc engram_store_load(engram_store **out, const char *path, const engram_key *key)
{
    uint8_t *p = NULL;
    size_t pn = 0;
    engram_rc rc;
    if (out) *out = NULL;
    if (!out || !path) return ENGRAM_E_ARG;
    rc = engram_seal_read(path, ENGRAM_KIND_STORE, key, &p, &pn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_store_deserialize(out, p, pn);
    engram_wipe(p, pn);
    engram_free(p);
    return rc;
}
