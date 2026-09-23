/* ==================================================================================================
 * test_store.c -- P1.3 (part 2): the episodic store's contract, proven.
 * ==================================================================================================
 *   S1  contract: refusals, empty store, text with no features
 *   S2  round trip: stored text is the chunk verbatim; ids consecutive from 1; chunks cover the text
 *   S3  ALL OR NOTHING: every allocation of an add is failed in turn, in three states -- a fresh store,
 *       a store that compacts first, a store that must evict -- and the store's fingerprint and
 *       counters are required unchanged after each failure, and equal to a clean run after the retry
 *   S4  delete: tombstoned, unfindable, never recalled, id never reused
 *   S5  capacity: refusal when nothing is consolidated; eviction of the OLDEST consolidated only;
 *       unconsolidated memories survive every eviction
 *   S6  compaction: logically invisible; tombstones stay bounded under endless evicting adds
 *   S7  determinism: two stores fed the same calls have the same fingerprint (STOREPRINT for the gate)
 *   S8  recall works end to end: a store of the English corpus finds the source of fragments
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_store.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"

#include <stdlib.h>
#include <string.h>

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

typedef struct { uint8_t *buf; const char *line[1024]; size_t len[1024]; size_t n; } lines;

/* one_ep: keep only paragraphs that fit ONE episode, for tests whose arithmetic counts episodes */
static int load_lines_f(lines *L, const char *lang, size_t maxn, int one_ep);
static int load_lines(lines *L, const char *lang, size_t maxn) { return load_lines_f(L, lang, maxn, 0); }
static int load_short(lines *L, const char *lang, size_t maxn) { return load_lines_f(L, lang, maxn, 1); }

static int load_lines_f(lines *L, const char *lang, size_t maxn, int one_ep)
{
    char name[64], *path;
    size_t blen = 0, i, s;
    memset(L, 0, sizeof *L);
    snprintf(name, sizeof name, "corpus_%s.txt", lang);
    path = engram_path_join(data_dir(), name);
    if (!path || engram_file_read(path, &L->buf, &blen) != ENGRAM_OK) { engram_free(path); return 0; }
    engram_free(path);
    for (i = 0; i < blen && L->n < maxn && L->n < 1024u; ) {
        s = i;
        while (i < blen && L->buf[i] != '\n') i++;
        if (i > s && (!one_ep || i - s <= ENGRAM_EPI_TAIL)) {
            L->line[L->n] = (const char *)L->buf + s; L->len[L->n] = i - s; L->n++;
        }
        i++;
    }
    return L->n > 0;
}

static engram_store *open_small(size_t max_ep, size_t max_bytes)
{
    engram_store_cfg c;
    engram_store *s = NULL;
    engram_store_cfg_default(&c);
    c.max_episodes = max_ep;
    c.max_text_bytes = max_bytes;
    if (engram_store_open(&s, &c) != ENGRAM_OK) return NULL;
    return s;
}

/* ---- S1, S2 ---------------------------------------------------------------------------------- */
static void test_contract(void)
{
    engram_store *s = NULL;
    engram_store_cfg c;
    engram_episode ep;
    engram_hit h[4];
    uint64_t first = 7;
    size_t nadd = 7, nh = 7;
    uint64_t fp;

    ET_SECTION("S1: refusals");
    ET_RC(engram_store_open(NULL, NULL), ENGRAM_E_ARG);
    engram_store_cfg_default(&c); c.max_episodes = 0;
    ET_RC(engram_store_open(&s, &c), ENGRAM_E_ARG);
    engram_store_cfg_default(&c); c.enc.tf = ENGRAM_TF_LINEAR;              /* no exact stage */
    ET_RC(engram_store_open(&s, &c), ENGRAM_E_ARG);
    engram_store_cfg_default(&c); c.chunk_cap = ENGRAM_EPI_TAIL + 1u;
    ET_RC(engram_store_open(&s, &c), ENGRAM_E_ARG);
    engram_store_cfg_default(&c); c.recall_c = 0;
    ET_RC(engram_store_open(&s, &c), ENGRAM_E_ARG);
    ET_CHECK(s == NULL);
    ET_OK(engram_store_open(&s, NULL));
    if (!s) return;
    ET_RC(engram_store_add(s, NULL, 3u, 0u, 0u, &first, &nadd), ENGRAM_E_ARG);
    ET_RC(engram_store_add(s, "abc", 3u, 0u, 0u, NULL, &nadd), ENGRAM_E_ARG);
    ET_RC(engram_store_get(s, 1u, &ep), ENGRAM_E_NOTFOUND);
    ET_RC(engram_store_delete(s, 1u), ENGRAM_E_NOTFOUND);
    ET_RC(engram_store_consolidated(s, 0u), ENGRAM_E_NOTFOUND);
    ET_OK(engram_store_recall(s, "anything at all", 15u, h, 4u, &nh));
    ET_EQ_U64(nh, 0u);
    ET_RC(engram_store_recall(s, "'' '", 4u, h, 4u, &nh), ENGRAM_E_SHORT);   /* apostrophes are IGNORE */

    ET_SECTION("S1: text with no encodable feature adds nothing and takes no id");
    fp = engram_store_fingerprint(s);
    ET_RC(engram_store_add(s, "   \n\t  ", 7u, 1u, 0u, &first, &nadd), ENGRAM_E_SHORT);
    ET_RC(engram_store_add(s, "'''", 3u, 1u, 0u, &first, &nadd), ENGRAM_E_SHORT);   /* IGNOREs only */
    ET_EQ_U64(nadd, 0u);
    ET_EQ_U64(engram_store_fingerprint(s), fp);

    ET_SECTION("S2: round trip -- the chunk verbatim, ids from 1, fields as given");
    ET_OK(engram_store_add(s, "The quick brown fox.", 20u, 1234u, 42u, &first, &nadd));
    ET_EQ_U64(first, 1u);
    ET_EQ_U64(nadd, 1u);
    ET_OK(engram_store_get(s, 1u, &ep));
    ET_CHECK(ep.len == 20u && memcmp(ep.text, "The quick brown fox.", 20u) == 0);
    ET_EQ_U64(ep.time_ms, 1234u);
    ET_EQ_U64(ep.source, 42u);
    ET_EQ_U64(ep.flags, ENGRAM_EPI_LIVE);
    ET_OK(engram_store_recall(s, "quick brown", 11u, h, 4u, &nh));
    ET_CHECK(nh == 1u && h[0].id == 1u && h[0].score > 0.0 && h[0].score <= 1.0);
    ET_OK(engram_store_get(s, 1u, &ep));
    ET_EQ_U64(ep.recalls, 1u);
    engram_store_close(s);

    ET_SECTION("S2: a long text becomes consecutive episodes that cover every non-space byte");
    {
        lines L;
        char *big;
        size_t i, m = 0, got = 0;
        ET_CHECK(load_lines(&L, "en", 40u));
        big = (char *)engram_malloc(64u * 1024u);
        s = open_small(1000u, 1u << 20);
        ET_CHECK(big && s);
        if (big && s) {
            for (i = 0; i < L.n; i++) { memcpy(big + m, L.line[i], L.len[i]); m += L.len[i]; big[m++] = '\n'; big[m++] = '\n'; }
            ET_OK(engram_store_add(s, big, m, 5u, 1u, &first, &nadd));
            ET_CHECK(nadd > 20u);
            {   /* the non-space bytes of the episodes, in order, equal the non-space bytes of the text */
                size_t a = 0, b, e;
                int same = 1;
                for (b = 0; b < nadd && same; b++) {
                    ET_OK(engram_store_get(s, first + b, &ep));
                    ET_CHECK(ep.len <= ENGRAM_EPI_TAIL);
                    for (e = 0; e < ep.len; e++) {
                        char ch = ep.text[e];
                        if (ch == ' ' || ch == '\n') continue;
                        while (a < m && (big[a] == ' ' || big[a] == '\n')) a++;
                        if (a >= m || big[a] != ch) { same = 0; break; }
                        a++; got++;
                    }
                }
                while (a < m && (big[a] == ' ' || big[a] == '\n')) a++;
                ET_CHECKF(same && a == m, "episodes do not reproduce the text (%lu of %lu bytes)",
                          (unsigned long)a, (unsigned long)m);
            }
        }
        engram_free(big);
        engram_store_close(s);
        engram_free(L.buf);
        (void)got;
    }
}

/* ---- S3: all or nothing ---------------------------------------------------------------------- */
typedef void (*prefix_fn)(engram_store *s, const lines *L);

static void prefix_fresh(engram_store *s, const lines *L) { (void)s; (void)L; }

static void prefix_tombstones(engram_store *s, const lines *L)
{
    uint64_t f; size_t k, i;
    for (i = 0; i < 12u; i++) (void)engram_store_add(s, L->line[i], L->len[i], i, 0u, &f, &k);
    for (i = 1; i <= 12u; i += 2u) (void)engram_store_delete(s, i);       /* six tombstones */
}

/* Exactly 16 episodes in a 16-episode store: the arrays grow 8 -> 16 and sit full, so the evicting
 * add below must ALSO grow them -- otherwise a failed allocation could never interrupt an eviction
 * and the sweep would prove nothing about it (the vacuity guard in sweep() caught exactly that). */
static void prefix_full(engram_store *s, const lines *L)
{
    uint64_t f; size_t k, i;
    for (i = 0; i < 16u; i++) (void)engram_store_add(s, L->line[i], L->len[i], i, 0u, &f, &k);
    for (i = 1; i <= 4u; i++) (void)engram_store_consolidated(s, i);      /* four may go */
}

static void sweep(const char *what, prefix_fn prefix, size_t max_ep, const lines *L, const char *t, size_t tn)
{
    engram_store *s;
    engram_store_stats st0, st1;
    uint64_t f0, fok, first;
    size_t nadd, i, m, bad_rc = 0, bad_fp = 0, bad_retry = 0;
    uint64_t calls;
    engram_rc rc;

    /* the clean run: how many allocations the add makes, and the state it leaves */
    s = open_small(max_ep, 1u << 20);
    if (!s) { ET_CHECK(0); return; }
    prefix(s, L);
    calls = engram_alloc_calls();
    ET_OK(engram_store_add(s, t, tn, 99u, 7u, &first, &nadd));
    m = (size_t)(engram_alloc_calls() - calls);
    fok = engram_store_fingerprint(s);
    engram_store_close(s);

    for (i = 1; i <= m; i++) {
        s = open_small(max_ep, 1u << 20);
        if (!s) { ET_CHECK(0); return; }
        prefix(s, L);
        f0 = engram_store_fingerprint(s);
        engram_store_stats_get(s, &st0);
        engram_alloc_fail_at(i);
        rc = engram_store_add(s, t, tn, 99u, 7u, &first, &nadd);
        engram_alloc_fail_at(0);
        engram_store_stats_get(s, &st1);
        if (rc != ENGRAM_E_MEM || nadd != 0u) bad_rc++;
        if (engram_store_fingerprint(s) != f0 || st1.live != st0.live || st1.next_id != st0.next_id ||
            st1.evictions != st0.evictions || st1.consolidated != st0.consolidated) bad_fp++;
        if (engram_store_add(s, t, tn, 99u, 7u, &first, &nadd) != ENGRAM_OK ||
            engram_store_fingerprint(s) != fok) bad_retry++;
        engram_store_close(s);
    }
    printf("       %-28s %lu allocation points, each failed in turn\n", what, (unsigned long)m);
    ET_CHECKF(m > 0u, "%s: the add made no allocation -- the sweep would prove nothing", what);
    ET_CHECKF(bad_rc == 0u, "%s: %lu failures did not return E_MEM", what, (unsigned long)bad_rc);
    ET_CHECKF(bad_fp == 0u, "%s: %lu failures CHANGED the store", what, (unsigned long)bad_fp);
    ET_CHECKF(bad_retry == 0u, "%s: %lu retries did not reach the clean state", what, (unsigned long)bad_retry);
}

static void test_atomic(void)
{
    lines L;
    char *big;
    size_t i, m = 0;
    ET_SECTION("S3: all or nothing -- every allocation of an add failed in turn");
    if (!load_short(&L, "en", 200u)) { ET_CHECK(0); return; }
    big = (char *)engram_malloc(128u * 1024u);
    if (!big) { ET_CHECK(0); engram_free(L.buf); return; }
    for (i = 100; i < 160u; i++) { memcpy(big + m, L.line[i], L.len[i]); m += L.len[i]; big[m++] = '\n'; }
    sweep("fresh store", prefix_fresh, 100000u, &L, big, m);
    sweep("store that compacts first", prefix_tombstones, 100000u, &L, big, m);
    sweep("store that must evict", prefix_full, 16u, &L, L.line[150], L.len[150]);
    engram_free(big);
    engram_free(L.buf);
}

/* ---- S4, S5, S6 ------------------------------------------------------------------------------- */
static void test_lifecycle(void)
{
    lines L;
    engram_store *s;
    engram_episode ep;
    engram_hit h[8];
    engram_store_stats st;
    uint64_t first, fp;
    size_t nadd, nh, i;

    if (!load_short(&L, "en", 400u)) { ET_CHECK(0); return; }

    ET_SECTION("S4: delete -- unfindable, never recalled, id never reused");
    s = open_small(1000u, 1u << 20);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    for (i = 0; i < 20u; i++) ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd));
    {
        char own[ENGRAM_EPI_TAIL];
        size_t olen;
        ET_OK(engram_store_get(s, 5u, &ep));
        olen = ep.len; memcpy(own, ep.text, olen);
        ET_OK(engram_store_delete(s, 5u));
        ET_RC(engram_store_delete(s, 5u), ENGRAM_E_NOTFOUND);
        ET_RC(engram_store_get(s, 5u, &ep), ENGRAM_E_NOTFOUND);
        ET_RC(engram_store_consolidated(s, 5u), ENGRAM_E_NOTFOUND);
        /* its own full text -- the strongest possible query -- must not bring it back */
        ET_OK(engram_store_recall(s, own, olen, h, 8u, &nh));
        ET_CHECK(nh > 0u);
        for (i = 0; i < nh; i++) ET_CHECK(h[i].id != 5u);
    }
    engram_store_stats_get(s, &st);
    ET_OK(engram_store_add(s, L.line[30], L.len[30], 30u, 0u, &first, &nadd));
    ET_EQ_U64(first, st.next_id);
    ET_EQ_U64(first, 21u);
    engram_store_close(s);

    ET_SECTION("S5: capacity -- refusal when nothing is consolidated");
    s = open_small(10u, 1u << 20);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    for (i = 0; i < 10u; i++) ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd));
    fp = engram_store_fingerprint(s);
    ET_RC(engram_store_add(s, L.line[10], L.len[10], 10u, 0u, &first, &nadd), ENGRAM_E_FULL);
    ET_EQ_U64(engram_store_fingerprint(s), fp);
    engram_store_stats_get(s, &st);
    ET_EQ_U64(st.refusals, 1u);

    ET_SECTION("S5: eviction takes the OLDEST CONSOLIDATED, and only as many as needed");
    ET_OK(engram_store_consolidated(s, 7u));
    ET_OK(engram_store_consolidated(s, 3u));
    ET_OK(engram_store_consolidated(s, 9u));
    ET_OK(engram_store_add(s, L.line[10], L.len[10], 10u, 0u, &first, &nadd));
    ET_RC(engram_store_get(s, 3u, &ep), ENGRAM_E_NOTFOUND);                    /* the oldest one */
    ET_OK(engram_store_get(s, 7u, &ep));
    ET_OK(engram_store_get(s, 9u, &ep));
    for (i = 1; i <= 10u; i++) if (i != 3u) ET_OK(engram_store_get(s, (uint64_t)i, &ep));
    engram_store_stats_get(s, &st);
    ET_EQ_U64(st.evictions, 1u);
    ET_EQ_U64(st.live, 10u);
    ET_OK(engram_store_add(s, L.line[11], L.len[11], 11u, 0u, &first, &nadd));
    ET_RC(engram_store_get(s, 7u, &ep), ENGRAM_E_NOTFOUND);
    ET_OK(engram_store_add(s, L.line[12], L.len[12], 12u, 0u, &first, &nadd));
    ET_RC(engram_store_get(s, 9u, &ep), ENGRAM_E_NOTFOUND);
    ET_RC(engram_store_add(s, L.line[13], L.len[13], 13u, 0u, &first, &nadd), ENGRAM_E_FULL);
    for (i = 1; i <= 10u; i++)                              /* every unconsolidated original survives */
        if (i != 3u && i != 7u && i != 9u) ET_OK(engram_store_get(s, (uint64_t)i, &ep));
    engram_store_close(s);

    ET_SECTION("S5: the text-byte cap is enforced the same way");
    s = open_small(1000u, 2000u);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    {
        size_t used = 0;
        for (i = 0; used + L.len[i] <= 2000u; i++) { ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd)); used += L.len[i]; }
        ET_RC(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd), ENGRAM_E_FULL);
        engram_store_stats_get(s, &st);
        ET_CHECK(st.text_bytes <= 2000u);
    }
    engram_store_close(s);

    ET_SECTION("S6: compaction is logically invisible, and tombstones stay bounded forever");
    s = open_small(50u, 1u << 20);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    for (i = 0; i < 50u; i++) ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd));
    for (i = 1; i <= 50u; i += 3u) ET_OK(engram_store_delete(s, (uint64_t)i));
    fp = engram_store_fingerprint(s);
    ET_OK(engram_store_recall(s, L.line[20], 40u, h, 8u, &nh));
    fp = engram_store_fingerprint(s);                     /* recall moves a counter: re-read */
    engram_store_compact(s);
    ET_EQ_U64(engram_store_fingerprint(s), fp);
    engram_store_stats_get(s, &st);
    ET_EQ_U64(st.tombstones, 0u);
    {
        size_t worst = 0, peak = 0;
        for (i = 0; i < 400u; i++) {                      /* a full store that evicts on every add */
            engram_store_stats sti;
            (void)engram_store_consolidated(s, first);
            if (engram_store_add(s, L.line[i % 400u], L.len[i % 400u], i, 0u, &first, &nadd) != ENGRAM_OK) {
                /* consolidate everything live and retry: the store must then make room */
                uint64_t id;
                engram_store_stats_get(s, &sti);
                for (id = 1; id < sti.next_id; id++) (void)engram_store_consolidated(s, id);
                ET_OK(engram_store_add(s, L.line[i % 400u], L.len[i % 400u], i, 0u, &first, &nadd));
            }
            engram_store_stats_get(s, &sti);
            if (sti.tombstones > worst) worst = sti.tombstones;
            if (sti.bytes_resident > peak) peak = sti.bytes_resident;
        }
        engram_store_stats_get(s, &st);
        printf("       400 evicting adds: live %lu, worst tombstones %lu, peak resident %lu bytes\n",
               (unsigned long)st.live, (unsigned long)worst, (unsigned long)peak);
        ET_CHECK(worst <= st.live / 4u + 64u + 1u);
        ET_CHECK(st.live <= 50u);
    }
    engram_store_close(s);
    engram_free(L.buf);
}

/* ---- S7, S8 ---------------------------------------------------------------------------------- */
static void test_determinism_and_recall(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    engram_store *a, *b;
    uint64_t first, fa, fb;
    size_t nadd, i, li;
    engram_hit h[4];
    size_t nh;

    ET_SECTION("S7: two stores fed the same calls are identical; the print for the gate");
    a = open_small(100000u, 64u << 20);
    b = open_small(100000u, 64u << 20);
    if (!a || !b) { ET_CHECK(0); engram_store_close(a); engram_store_close(b); return; }
    for (li = 0; li < 5u; li++) {
        lines L;
        if (!load_lines(&L, langs[li], 1024u)) { ET_CHECK(0); continue; }
        for (i = 0; i < L.n; i++) {
            (void)engram_store_add(a, L.line[i], L.len[i], i, (uint32_t)li, &first, &nadd);
            (void)engram_store_add(b, L.line[i], L.len[i], i, (uint32_t)li, &first, &nadd);
            if (i % 7u == 3u) { (void)engram_store_delete(a, first); (void)engram_store_delete(b, first); }
            if (i % 5u == 1u) {
                (void)engram_store_recall(a, L.line[i], L.len[i] / 5u + 8u, h, 4u, &nh);
                (void)engram_store_recall(b, L.line[i], L.len[i] / 5u + 8u, h, 4u, &nh);
            }
        }
        engram_free(L.buf);
    }
    fa = engram_store_fingerprint(a);
    fb = engram_store_fingerprint(b);
    ET_EQ_U64(fa, fb);
    printf("STOREPRINT %016llX\n", (unsigned long long)fa);
    engram_store_close(a);
    engram_store_close(b);

    ET_SECTION("S8: recall end to end -- fragments of English paragraphs find their episode");
    {
        lines L;
        engram_store *s = open_small(100000u, 64u << 20);
        size_t hits = 0, total = 0;
        uint64_t *idof;
        if (!s || !load_lines(&L, "en", 1024u)) { ET_CHECK(0); engram_store_close(s); return; }
        idof = (uint64_t *)engram_array(L.n, sizeof *idof);
        for (i = 0; i < L.n && idof; i++) {
            ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd));
            idof[i] = nadd == 1u ? first : 0u;             /* paragraphs under 480 bytes are one episode */
        }
        for (i = 0; i < L.n && idof; i += 3u) {
            size_t q0 = L.len[i] / 3u, ql = L.len[i] / 5u;  /* a fifth of it, from a third of the way in */
            if (!idof[i] || ql < 12u) continue;
            while (q0 > 0u && L.line[i][q0 - 1u] != ' ') q0--;
            if (engram_store_recall(s, L.line[i] + q0, ql, h, 4u, &nh) == ENGRAM_OK && nh > 0u && h[0].id == idof[i]) hits++;
            total++;
        }
        printf("       %lu of %lu fragments found their episode first (%.4f)\n", (unsigned long)hits,
               (unsigned long)total, total ? (double)hits / (double)total : 0.0);
        ET_CHECK(total > 100u && hits >= total * 95u / 100u);
        engram_free(idof);
        engram_free(L.buf);
        engram_store_close(s);
    }
}

int main(void)
{
    printf("ENGRAM P1.3 -- episodic store\n");
    ET_SELFTEST();
    test_contract();
    test_atomic();
    test_lifecycle();
    test_determinism_and_recall();
    return et_report("test_store");
}
