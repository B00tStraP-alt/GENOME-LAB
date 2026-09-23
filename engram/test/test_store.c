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
 *   S8  RECALL AT SCALE: every paragraph of the scale corpus (26,139 episodes), fragments of 12-96 bytes
 *       with 0-2 typos and keyword controls; the three rankings on the same candidates; FULL beats BAG
 *       beats EXACT on fragments and costs the control at most 1%; every answer checked against the
 *       ranking contract; stores opened with the lesion ranks agree; floors; RECALLPRINT for the gate
 *   S9  the ranking's contract by hand: the single-candidate rule, alignment ordering, a cue too long to
 *       align, and open/recall under allocation failure
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_store.h"
#include "../src/engram_align.h"
#include "../src/engram_chunk.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "../src/engram_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* S8 floors: the values measured on S8's protocol minus two standard errors, rounded down.
 *   full  (100 sources, 1,800 fragment cues, 399 control cues)  fragments 0.9283 (se 0.0061)  control 0.9223 (se 0.0134)
 *   quick ( 40 sources,   720 fragment cues, 160 control cues)  fragments 0.9389 (se 0.0089)  control 0.9250 (se 0.0208)
 * The computation is deterministic (RECALLPRINT is compared across platforms by the gate), so a value
 * under its floor means the CODE changed, not the luck. */
#define FLOOR_FRAG   0.91
#define FLOOR_CTRL   0.89
#define FLOOR_FRAG_Q 0.92
#define FLOOR_CTRL_Q 0.88

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
    engram_store_cfg_default(&c); c.rank = (engram_rank)3;
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
    ET_CHECK(nh == 1u && h[0].id == 1u && h[0].exact > 0.0 && h[0].exact <= 1.0 && h[0].contain > 0.0 &&
             h[0].contain <= 1.0);
    ET_CHECK(h[0].edits == 0u && h[0].aligned == 1u);    /* verbatim, and the only candidate */
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

    ET_SECTION("S5: eviction takes CONSOLIDATED episodes only, least recalled first (oldest among equals),"
               " and only as many as needed");
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
    {   /* recall episode 7 by its own text: now it is the MORE recalled, so 9 goes before it */
        char own[ENGRAM_EPI_TAIL];
        size_t olen;
        ET_OK(engram_store_get(s, 7u, &ep));
        olen = ep.len; memcpy(own, ep.text, olen);
        ET_OK(engram_store_recall(s, own, olen, h, 8u, &nh));
        ET_CHECK(nh > 0u && h[0].id == 7u);
        ET_OK(engram_store_get(s, 7u, &ep));
        ET_EQ_U64(ep.recalls, 1u);
    }
    ET_OK(engram_store_add(s, L.line[11], L.len[11], 11u, 0u, &first, &nadd));
    ET_RC(engram_store_get(s, 9u, &ep), ENGRAM_E_NOTFOUND);            /* 0 recalls, younger than 3 */
    ET_OK(engram_store_get(s, 7u, &ep));                                /* 1 recall: kept */
    ET_OK(engram_store_add(s, L.line[12], L.len[12], 12u, 0u, &first, &nadd));
    ET_RC(engram_store_get(s, 7u, &ep), ENGRAM_E_NOTFOUND);            /* the last consolidated one */
    ET_RC(engram_store_add(s, L.line[13], L.len[13], 13u, 0u, &first, &nadd), ENGRAM_E_FULL);
    for (i = 1; i <= 10u; i++)                              /* every unconsolidated original survives */
        if (i != 3u && i != 7u && i != 9u) ET_OK(engram_store_get(s, (uint64_t)i, &ep));
    engram_store_close(s);

    ET_SECTION("S5: recall counts past the histogram (63+) -- the least recalled still goes first");
    s = open_small(4u, 1u << 20);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    {
        static const unsigned want[4] = { 70u, 65u, 80u, 66u };   /* episode 2 is the least recalled */
        unsigned e, r;
        for (i = 0; i < 4u; i++) ET_OK(engram_store_add(s, L.line[i], L.len[i], i, 0u, &first, &nadd));
        for (e = 0; e < 4u; e++) {
            char own[ENGRAM_EPI_TAIL];
            size_t olen;
            ET_OK(engram_store_get(s, e + 1u, &ep));
            olen = ep.len; memcpy(own, ep.text, olen);
            for (r = 0; r < want[e]; r++) (void)engram_store_recall(s, own, olen, h, 1u, &nh);
            ET_OK(engram_store_get(s, e + 1u, &ep));
            ET_EQ_U64(ep.recalls, want[e]);
            ET_OK(engram_store_consolidated(s, e + 1u));
        }
        ET_OK(engram_store_add(s, L.line[4], L.len[4], 4u, 0u, &first, &nadd));
        ET_RC(engram_store_get(s, 2u, &ep), ENGRAM_E_NOTFOUND);
        for (e = 1; e <= 4u; e++) if (e != 2u) ET_OK(engram_store_get(s, e, &ep));
        ET_OK(engram_store_add(s, L.line[5], L.len[5], 5u, 0u, &first, &nadd));
        ET_RC(engram_store_get(s, 4u, &ep), ENGRAM_E_NOTFOUND);        /* 66: next least recalled */
    }
    engram_store_close(s);

    ET_SECTION("S5: the text-byte cap is enforced the same way -- a big text takes the least recalled, as many as needed");
    s = open_small(1000u, 1000u);
    if (!s) { ET_CHECK(0); engram_free(L.buf); return; }
    {
        char ep_text[11][120], big[170];                /* a word may overshoot the length by 4 */
        uint64_t e;
        size_t m;
        for (e = 0; e < 11u; e++) {                     /* ten 100-byte episodes, and a 150-byte text */
            char *t = e < 10u ? ep_text[e] : big;
            size_t want = e < 10u ? 100u : 150u, w = 0;
            unsigned k = 0;
            while (w < want) {                          /* distinct words per episode: e and k spelled */
                if (w) t[w++] = ' ';
                t[w++] = (char)('a' + (e * 7u + k) % 26u); t[w++] = (char)('a' + (e * 3u + k * 5u) % 26u);
                t[w++] = (char)('a' + (k * 11u + e) % 26u); t[w++] = (char)('a' + (e + k) % 26u);
                k++;
            }
            t[want - 1u] = 'z';                         /* no trailing space: the episode is exactly want */
            t[want] = 0;
        }
        for (e = 0; e < 10u; e++) {
            ET_OK(engram_store_add(s, ep_text[e], 100u, e, 0u, &first, &nadd));
            ET_EQ_U64(nadd, 1u);
        }
        engram_store_stats_get(s, &st);
        ET_EQ_U64(st.text_bytes, 1000u);
        for (e = 1; e <= 10u; e++) ET_OK(engram_store_consolidated(s, e));
        for (e = 1; e <= 10u; e++) {                    /* recall every episode but 2 and 3 */
            if (e == 2u || e == 3u) continue;
            ET_OK(engram_store_recall(s, ep_text[e - 1u], 100u, h, 1u, &nh));
            ET_CHECK(nh == 1u && h[0].id == e);
        }
        m = strlen(big);
        ET_OK(engram_store_add(s, big, m, 99u, 0u, &first, &nadd));   /* needs 150: two go, not one */
        ET_RC(engram_store_get(s, 2u, &ep), ENGRAM_E_NOTFOUND);
        ET_RC(engram_store_get(s, 3u, &ep), ENGRAM_E_NOTFOUND);
        for (e = 1; e <= 10u; e++) if (e != 2u && e != 3u) ET_OK(engram_store_get(s, e, &ep));
        engram_store_stats_get(s, &st);
        ET_EQ_U64(st.evictions, 2u);
        ET_EQ_U64(st.text_bytes, 950u);
    }
    engram_store_close(s);

    ET_SECTION("S5: the text-byte cap refuses when nothing is consolidated");
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

}

/* ---- S9: the ranking's contract -------------------------------------------------------------- */
static int ranks_before(const engram_hit *a, const engram_hit *b, engram_rank mode)
{
    if (mode == ENGRAM_RANK_FULL) {
        if (a->aligned != b->aligned) return a->aligned > b->aligned;
        if (a->aligned && a->edits != b->edits) return a->edits < b->edits;
    }
    if (mode != ENGRAM_RANK_EXACT && a->contain != b->contain) return a->contain > b->contain;
    if (a->exact != b->exact) return a->exact > b->exact;
    return a->id < b->id;
}

/* The significance rule of THE RANKING, recomputed from its definition -- sorted copy, not a
 * histogram -- over an answer that holds EVERY candidate (k >= C, or k >= live). */
static unsigned aligned_rule_ok(const engram_hit *h, size_t nh, size_t qcp)
{
    uint32_t e[256];
    size_t i, j;
    double best = 0.0;
    uint64_t med2;
    unsigned bad = 0;
    if (nh == 0u || nh > 256u) return nh > 256u;
    if (h[0].edits == ENGRAM_EDITS_NONE) {          /* no alignment: none computed, none claimed */
        for (i = 0; i < nh; i++) bad += h[i].edits != ENGRAM_EDITS_NONE || h[i].aligned;
        return bad;
    }
    for (i = 0; i < nh; i++) {
        uint32_t v = h[i].edits;
        for (j = i; j > 0u && e[j - 1u] > v; j--) e[j] = e[j - 1u];
        e[j] = v;
        if (h[i].contain > best) best = h[i].contain;
    }
    med2 = (uint64_t)e[(nh - 1u) / 2u] + e[nh / 2u];
    for (i = 0; i < nh; i++) {
        uint64_t ed = h[i].edits;
        unsigned want = 6u * ed <= med2 && 3u * ed <= (uint64_t)qcp && h[i].contain >= best - 0.2;
        bad += want != h[i].aligned;
    }
    return bad;
}

/* the cue's length as the alignment counts it; (size_t)-1 when it is too long to hold */
static size_t cue_cps(const engram_store_cfg *cfg, const char *q, size_t n)
{
    static uint32_t tmp[1u << 16];
    size_t len = 0;
    if (engram_align_norm(q, n, cfg->enc.fold, cfg->enc.utf8, tmp, sizeof tmp / sizeof *tmp, &len) != ENGRAM_OK)
        return (size_t)-1;
    return len;
}

/* every documented property of one answer; returns the number of violations */
static unsigned hits_ok(const engram_hit *h, size_t nh, engram_rank mode, size_t qcp)
{
    unsigned bad = 0;
    size_t i;
    for (i = 0; i < nh; i++) {
        if (!(h[i].contain >= 0.0 && h[i].contain <= 1.0 && h[i].exact >= 0.0 && h[i].exact <= 1.0)) bad++;
        if (mode != ENGRAM_RANK_FULL && (h[i].edits != ENGRAM_EDITS_NONE || h[i].aligned)) bad++;
        if (h[i].aligned && (h[i].edits == ENGRAM_EDITS_NONE || 3u * (size_t)h[i].edits > qcp)) bad++;
        if (h[i].edits != ENGRAM_EDITS_NONE && h[i].edits > qcp) bad++;
        if (i > 0u && !ranks_before(&h[i - 1u], &h[i], mode)) bad++;
    }
    return bad;
}

static engram_store *open_rank(engram_rank mode, size_t max_ep)
{
    engram_store_cfg c;
    engram_store *s = NULL;
    engram_store_cfg_default(&c);
    c.max_episodes = max_ep;
    c.rank = mode;
    if (engram_store_open(&s, &c) != ENGRAM_OK) return NULL;
    return s;
}

static void test_ranking(void)
{
    static const char *three[3] = {
        "the quick brown fox jumps over the lazy dog",
        "a slow green turtle crawls under the busy bridge",
        "brown quick jumps fox the lazy over dog",
    };
    engram_store *s;
    engram_hit h[8];
    uint64_t first, fp;
    size_t nadd, nh, i, m;
    engram_rank mode;

    ET_SECTION("S9: one candidate -- its alignment is significant only if it is verbatim");
    s = open_rank(ENGRAM_RANK_FULL, 100u);
    if (!s) { ET_CHECK(0); return; }
    ET_OK(engram_store_add(s, "the quick brown fox jumps", 25u, 0u, 0u, &first, &nadd));
    ET_OK(engram_store_recall(s, "quick brown", 11u, h, 8u, &nh));
    ET_CHECK(nh == 1u && h[0].edits == 0u && h[0].aligned == 1u);
    ET_OK(engram_store_recall(s, "quikc brown", 11u, h, 8u, &nh));    /* 1 edit, median 1: 3 > 1 */
    ET_CHECK(nh == 1u && h[0].edits == 1u && h[0].aligned == 0u);
    engram_store_close(s);

    ET_SECTION("S9: two candidates -- the median is the MEAN of the middle two (edits 1 and b: aligned iff 5 <= b)");
    s = open_rank(ENGRAM_RANK_FULL, 100u);
    if (!s) { ET_CHECK(0); return; }
    ET_OK(engram_store_add(s, three[0], strlen(three[0]), 0u, 0u, &first, &nadd));
    ET_OK(engram_store_add(s, three[1], strlen(three[1]), 1u, 0u, &first, &nadd));
    ET_OK(engram_store_recall(s, "quick brwn fox", 14u, h, 8u, &nh));
    ET_CHECK(nh == 2u);
    ET_CHECKF(nh == 2u && h[0].id == 1u && h[0].edits == 1u && h[1].edits >= 5u && h[0].aligned == 1u,
              "edits %u and %u, aligned %u", h[0].edits, nh == 2u ? h[1].edits : 0u, h[0].aligned);
    ET_EQ_U64(aligned_rule_ok(h, nh, 14u), 0u);
    engram_store_close(s);

    ET_SECTION("S9: the alignment puts the episode that holds the cue IN ORDER first; BAG and EXACT do not align");
    for (mode = ENGRAM_RANK_FULL; mode <= ENGRAM_RANK_EXACT; mode = (engram_rank)(mode + 1)) {
        s = open_rank(mode, 100u);
        if (!s) { ET_CHECK(0); return; }
        for (i = 0; i < 3u; i++) ET_OK(engram_store_add(s, three[i], strlen(three[i]), i, 0u, &first, &nadd));
        ET_OK(engram_store_recall(s, "quick brwn fox", 14u, h, 8u, &nh));
        ET_CHECK(nh == 3u);
        ET_EQ_U64(hits_ok(h, nh, mode, 14u), 0u);
        if (mode == ENGRAM_RANK_FULL) ET_EQ_U64(aligned_rule_ok(h, nh, 14u), 0u);
        if (mode == ENGRAM_RANK_FULL) {
            ET_CHECKF(h[0].id == 1u && h[0].aligned == 1u && h[0].edits == 1u,
                      "first %llu aligned %u edits %u", (unsigned long long)h[0].id, h[0].aligned, h[0].edits);
            for (i = 1; i < nh; i++) ET_CHECK(h[i].aligned == 0u && h[i].edits > 1u);
        }
        engram_store_close(s);
    }

    ET_SECTION("S9: a cue too long to align with ANY episode is ranked by the bag, and says so");
    s = open_rank(ENGRAM_RANK_FULL, 100u);
    if (!s) { ET_CHECK(0); return; }
    {
        /* Longer than 3/2 of the longest episode the store can hold AFTER folding: ENGRAM_FOLD_OUT_MAX
         * codepoints per byte, 4 x 480 x 3/2 = 2880. No episode could align with it (edits >= |q| - |d|
         * > |q| / 3), so none is computed. */
        static char longq[3200];
        engram_hit hb[8];
        engram_store *b = open_rank(ENGRAM_RANK_BAG, 100u);
        for (i = 0; i + 1u < sizeof longq; i++) longq[i] = (char)("abcdefghij klmnop"[i % 17u]);
        m = sizeof longq - 1u;
        ET_CHECK(b != NULL);
        for (i = 0; i < 3u && b; i++) {
            ET_OK(engram_store_add(s, three[i], strlen(three[i]), i, 0u, &first, &nadd));
            ET_OK(engram_store_add(b, three[i], strlen(three[i]), i, 0u, &first, &nadd));
        }
        ET_OK(engram_store_add(s, longq, 400u, 9u, 0u, &first, &nadd));
        if (b) ET_OK(engram_store_add(b, longq, 400u, 9u, 0u, &first, &nadd));
        ET_OK(engram_store_recall(s, longq, m, h, 8u, &nh));
        if (b) ET_OK(engram_store_recall(b, longq, m, hb, 8u, &i));
        ET_CHECK(nh > 0u && b && i == nh);
        for (i = 0; i < nh; i++) ET_CHECK(h[i].edits == ENGRAM_EDITS_NONE && h[i].aligned == 0u);
        for (i = 0; b && i < nh; i++) ET_CHECK(h[i].id == hb[i].id);
        engram_store_close(b);
    }
    engram_store_close(s);

    ET_SECTION("S9: open and recall under allocation failure -- E_MEM, nothing leaked, nothing changed");
    {
        size_t bad = 0, points;
        uint64_t calls;
        engram_store_cfg c;
        engram_store_cfg_default(&c);
        calls = engram_alloc_calls();
        s = NULL;
        ET_OK(engram_store_open(&s, &c));
        points = (size_t)(engram_alloc_calls() - calls);
        engram_store_close(s);
        for (i = 1; i <= points; i++) {
            s = (engram_store *)&bad;               /* must be reset to NULL on failure */
            engram_alloc_fail_at(i);
            if (engram_store_open(&s, &c) != ENGRAM_E_MEM || s != NULL) bad++;
            engram_alloc_fail_at(0);
            if (s && s != (engram_store *)&bad) engram_store_close(s);
        }
        printf("       open: %lu allocation points, each failed in turn\n", (unsigned long)points);
        ET_CHECK(points >= 5u);
        ET_EQ_U64(bad, 0u);
        s = open_rank(ENGRAM_RANK_FULL, 100u);
        if (!s) { ET_CHECK(0); return; }
        for (i = 0; i < 3u; i++) ET_OK(engram_store_add(s, three[i], strlen(three[i]), i, 0u, &first, &nadd));
        fp = engram_store_fingerprint(s);
        calls = engram_alloc_calls();
        engram_alloc_fail_at(1);
        ET_RC(engram_store_recall(s, "lazy dog", 8u, h, 8u, &nh), ENGRAM_E_MEM);
        engram_alloc_fail_at(0);
        ET_EQ_U64(nh, 0u);
        ET_EQ_U64(engram_store_fingerprint(s), fp);           /* no recall counter moved */
        engram_alloc_fail_at(2);
        ET_RC(engram_store_recall(s, "lazy dog", 8u, h, 8u, &nh), ENGRAM_E_MEM);
        engram_alloc_fail_at(0);
        ET_EQ_U64(engram_store_fingerprint(s), fp);
        ET_OK(engram_store_recall(s, "lazy dog", 8u, h, 8u, &nh));
        ET_CHECK(engram_alloc_calls() - calls >= 2u);
        ET_CHECK(engram_store_fingerprint(s) != fp);         /* now one did */
        engram_store_close(s);
    }
}

/* ---- S8: RECALL AT SCALE ----------------------------------------------------------------------
 * The P1.3 end-to-end protocol (lab e2e.c), through this API and nothing else:
 *   store      every paragraph of scale_docs.txt, added whole -- the store cuts them (about 26,000
 *              episodes of en/fr/de/sv/zh)
 *   sources    paragraph d = (t * 7919 + 13) mod D for t = 0, 1, 2, ..., skipping paragraphs under 120
 *              bytes; nsrc of them
 *   fragments  a cue of 12 / 16 / 24 / 40 / 64 / 96 bytes from a word start (any ideograph is one),
 *              x 0 / 1 / 2 typos (substitute / delete / insert / swap, letters from the paragraph).
 *              HIT: the first episode lies in the same paragraph and covers at least half the cue's
 *              bytes -- a cue may straddle two episodes, and either half-holder is the memory
 *   control    kw3 / kw5 / kw8: that many pieces (letter runs of 3+, ideograph pairs) of one episode, in
 *              random order; shuf: the pieces of a 15% window, shuffled. No contiguous text: the cue
 *              the alignment must not hurt. HIT: the first episode holds every piece
 * One FULL recall with k = C per cue sees every candidate with its three measurements, so BAG and
 * EXACT are the same candidates re-ordered by the test (and a subsample proves stores OPENED with
 * those ranks agree). Floors are the measured values minus two standard errors. */
typedef struct { size_t doc, off, len; } span_t;

static int letter(uint32_t cp)
{
    unsigned k = engram_cp_class(cp);
    return k == ENGRAM_CLASS_WORD || k == ENGRAM_CLASS_IDEO;
}

static size_t dec_at(const char *t, size_t n, uint32_t *cp, size_t *at)
{
    size_t o = 0, k = 0;
    while (o < n) { int v; at[k] = o; o += engram_utf8_decode((const uint8_t *)t + o, n - o, &cp[k], &v); k++; }
    at[k] = n;
    return k;
}

static size_t enc_cps(const uint32_t *cp, size_t n, char *out)
{
    size_t i, k = 0;
    for (i = 0; i < n; i++) k += engram_utf8_encode(cp[i], (uint8_t *)out + k);
    return k;
}

static void rerank(const engram_hit *h, size_t n, uint64_t first[3])
{
    size_t i, b = 0, e = 0;
    for (i = 1; i < n; i++) {
        if (ranks_before(&h[i], &h[b], ENGRAM_RANK_BAG)) b = i;
        if (ranks_before(&h[i], &h[e], ENGRAM_RANK_EXACT)) e = i;
    }
    first[0] = h[0].id; first[1] = h[b].id; first[2] = h[e].id;
}

#define SC_LENS 6
#define SC_FRAG (SC_LENS * 3)
#define SC_CTRL 4

static void test_scale(int quick)
{
    static const size_t LENS[SC_LENS] = { 12u, 16u, 24u, 40u, 64u, 96u };
    static const char *CTRL[SC_CTRL] = { "kw3", "kw5", "kw8", "shuf" };
    static uint32_t cp[65536], out[65536];
    static size_t at[65537], starts[65536];
    static char buf[65536];
    char *path;
    uint8_t *raw = NULL;
    size_t rawn = 0, i, nd = 0, nsp = 0, t, got = 0, nsrc = quick ? 40u : 100u, C, qc = 0;
    const char **D = NULL;
    size_t *DL = NULL;
    span_t *SP = NULL;
    engram_store *s = NULL, *lesion[2] = { NULL, NULL };
    engram_store_cfg cfg;
    engram_hit *h = NULL, *hl = NULL;
    double t0, ms;
    unsigned fr[SC_FRAG][3], fn[SC_FRAG], cr[SC_CTRL][3], cn[SC_CTRL], cand[2] = { 0, 0 };
    unsigned win[2] = { 0, 0 }, loss[2] = { 0, 0 }, badhits = 0, lesion_n = 0, lesion_bad = 0;
    unsigned shuffled = 0, shuffled_n = 0, a, r;
    uint64_t rp = 0x5245434Cull;

    memset(fr, 0, sizeof fr); memset(fn, 0, sizeof fn); memset(cr, 0, sizeof cr); memset(cn, 0, sizeof cn);
    ET_SECTION("S8: recall at scale -- the whole scale corpus, cut by the store");
    path = engram_path_join(data_dir(), "scale_docs.txt");
    if (!path || engram_file_read(path, &raw, &rawn) != ENGRAM_OK) { ET_CHECKF(0, "scale_docs.txt did not load"); engram_free(path); return; }
    engram_free(path);
    for (i = 0; i < rawn; i++) nd += raw[i] == '\n';
    D = (const char **)engram_array(nd + 1u, sizeof *D);
    DL = (size_t *)engram_array(nd + 1u, sizeof *DL);
    engram_store_cfg_default(&cfg);
    cfg.max_episodes = 200000u;
    ET_OK(engram_store_open(&s, &cfg));
    SP = (span_t *)engram_array(cfg.max_episodes, sizeof *SP);
    h = (engram_hit *)engram_array(cfg.recall_c, sizeof *h);
    hl = (engram_hit *)engram_array(cfg.recall_c, sizeof *hl);
    if (!D || !DL || !s || !SP || !h || !hl) { ET_CHECK(0); goto out; }
    C = cfg.recall_c;
    nd = 0;
    for (i = 0; i < rawn; ) {
        size_t e = i;
        while (e < rawn && raw[e] != '\n') e++;
        if (e > i) { D[nd] = (const char *)raw + i; DL[nd] = e - i; nd++; }
        i = e + 1u;
    }
    t0 = (double)engram_now_ns();
    for (i = 0; i < nd; i++) {                    /* add each paragraph; learn each episode's span */
        uint64_t first = 0, sg[ENGRAM_SIG_WORDS];
        size_t nadd = 0, off, len, k = 0;
        engram_chunkit it;
        if (engram_store_add(s, D[i], DL[i], i, 0u, &first, &nadd) != ENGRAM_OK) continue;
        (void)engram_chunkit_init(&it, D[i], DL[i], cfg.chunk_cap, cfg.chunk_overlap);
        while (engram_chunkit_next(&it, &off, &len))
            if (engram_encode_sig(&cfg.enc, D[i] + off, len, sg, NULL) == ENGRAM_OK && first - 1u + k < cfg.max_episodes) {
                SP[first - 1u + k].doc = i; SP[first - 1u + k].off = off; SP[first - 1u + k].len = len; k++;
            }
        if (k != nadd) lesion_bad++;
        nsp = first - 1u + k;
    }
    {
        engram_store_stats st;
        engram_store_stats_get(s, &st);
        printf("       %lu paragraphs -> %lu episodes (%lu text bytes, %.1f MB resident) in %.2f s\n",
               (unsigned long)nd, (unsigned long)st.live, (unsigned long)st.text_bytes,
               (double)st.bytes_resident / 1048576.0, ((double)engram_now_ns() - t0) / 1e9);
        ET_CHECK(st.live > 20000u && st.live == nsp);
        ET_EQ_U64(lesion_bad, 0u);                /* the test's spans are the store's episodes */
    }
    for (r = 0; r < 2u; r++) {                    /* the lesions, for the agreement check */
        engram_store_cfg lc = cfg;
        lc.rank = r ? ENGRAM_RANK_EXACT : ENGRAM_RANK_BAG;
        if (engram_store_open(&lesion[r], &lc) != ENGRAM_OK) { ET_CHECK(0); goto out; }
        for (i = 0; i < nd; i++) { uint64_t f; size_t k; (void)engram_store_add(lesion[r], D[i], DL[i], i, 0u, &f, &k); }
    }

    t0 = (double)engram_now_ns();
    for (t = 0; got < nsrc; t++) {
        size_t d = (t * 7919u + 13u) % nd, n, ns = 0, nal = 0, x;
        uint32_t al[4096];
        if (DL[d] < 120u || DL[d] > sizeof buf / 2u) continue;
        got++;
        n = dec_at(D[d], DL[d], cp, at);
        for (i = 0; i < n; i++) {
            if (letter(cp[i]) && nal < 4096u) al[nal++] = cp[i];
            if (engram_cp_class(cp[i]) == ENGRAM_CLASS_IDEO ||
                (letter(cp[i]) && (i == 0u || engram_cp_class(cp[i - 1u]) == ENGRAM_CLASS_SPACE))) starts[ns++] = i;
        }
        if (!nal) { al[0] = 'e'; nal = 1u; }
        for (a = 0; a < SC_FRAG; a++) {           /* ---- fragments ---- */
            engram_rng rg;
            size_t L = LENS[a % SC_LENS], typos = a / SC_LENS, pool = 0, pick, st0, en, qn, tl, nh = 0, cs, ce, k;
            uint64_t f[3];
            int ok[3], any = 0;
            engram_rng_seed(&rg, engram_mix2(0xE2E0u + a, (uint64_t)d));
            for (x = 0; x < ns; x++) if (at[starts[x]] + L <= DL[d]) pool++;
            if (!pool) continue;
            pick = (size_t)engram_rng_below(&rg, pool);
            for (x = 0; x < ns; x++) if (at[starts[x]] + L <= DL[d]) { if (!pick) break; pick--; }
            st0 = starts[x];
            for (en = st0; en < n && at[en + 1u] - at[st0] <= L; en++) ;
            qn = en - st0;
            memcpy(out, cp + st0, qn * sizeof *out);
            for (k = 0; k < typos; k++) {
                size_t p = 0, tries;
                unsigned op;
                for (tries = 0; tries < 64u; tries++) { p = (size_t)engram_rng_below(&rg, qn); if (letter(out[p])) break; }
                if (!letter(out[p])) break;
                op = (unsigned)engram_rng_below(&rg, 4u);
                if (op == 0u) out[p] = al[engram_rng_below(&rg, nal)];
                else if (op == 1u && qn > 1u) { memmove(out + p, out + p + 1u, (qn - p - 1u) * sizeof *out); qn--; }
                else if (op == 2u) { memmove(out + p + 1u, out + p, (qn - p) * sizeof *out); out[p] = al[engram_rng_below(&rg, nal)]; qn++; }
                else if (p + 1u < qn) { uint32_t tmp = out[p]; out[p] = out[p + 1u]; out[p + 1u] = tmp; }
            }
            tl = enc_cps(out, qn, buf);
            fn[a]++;
            if (engram_store_recall(s, buf, tl, h, C, &nh) != ENGRAM_OK || !nh) continue;
            qc = cue_cps(&cfg, buf, tl);
            badhits += hits_ok(h, nh, ENGRAM_RANK_FULL, qc) + aligned_rule_ok(h, nh, qc) + (nh != C);
            for (k = 0; k < nh; k++) {
                rp = engram_mix2(rp, h[k].id);
                rp = engram_mix2(rp, engram_hash_bytes(&h[k].contain, sizeof h[k].contain, h[k].edits));
                rp = engram_mix2(rp, engram_hash_bytes(&h[k].exact, sizeof h[k].exact, h[k].aligned));
            }
            rerank(h, nh, f);
            cs = at[st0]; ce = at[en];
            for (r = 0; r < 3u; r++) {
                span_t e = SP[f[r] - 1u];
                size_t lo = e.off > cs ? e.off : cs, hi = e.off + e.len < ce ? e.off + e.len : ce;
                ok[r] = e.doc == d && hi > lo && 2u * (hi - lo) >= ce - cs;
                fr[a][r] += (unsigned)ok[r];
            }
            for (k = 0; k < nh && !any; k++) {
                span_t e = SP[h[k].id - 1u];
                size_t lo = e.off > cs ? e.off : cs, hi = e.off + e.len < ce ? e.off + e.len : ce;
                any = e.doc == d && hi > lo && 2u * (hi - lo) >= ce - cs;
            }
            cand[0] += (unsigned)any;
            win[0] += ok[0] && !ok[1]; loss[0] += !ok[0] && ok[1];
            {   /* the instrument can fail: the same criterion against an unrelated paragraph's span */
                size_t od = (d + nd / 2u) % nd;
                span_t e = SP[f[0] - 1u];
                shuffled += e.doc == od; shuffled_n++;
            }
            if ((got + a) % 23u == 0u) {          /* lesion agreement: stores opened with BAG / EXACT */
                for (r = 0; r < 2u; r++) {
                    size_t nl = 0;
                    if (engram_store_recall(lesion[r], buf, tl, hl, C, &nl) != ENGRAM_OK || !nl ||
                        hl[0].id != f[r + 1u] || hits_ok(hl, nl, r ? ENGRAM_RANK_EXACT : ENGRAM_RANK_BAG, qc))
                        lesion_bad++;
                    lesion_n++;
                }
            }
        }
        {                                         /* ---- control ---- */
            size_t e0 = 0, e1 = 0;
            for (x = 0; x < nsp; x++) if (SP[x].doc == d) { if (!e1) e0 = x; e1 = x + 1u; }
            for (a = 0; e1 && a < SC_CTRL; a++) {
                engram_rng rg;
                size_t ep, pn = 0, lo = 0, hi, want, y, m = 0, tl, nh = 0, en2, k;
                static size_t ps[4096], pl[4096], perm[4096];
                uint64_t f[3];
                int ok[3];
                engram_rng_seed(&rg, engram_mix2(0xC7A1u + a, (uint64_t)d));
                ep = e0 + (size_t)engram_rng_below(&rg, e1 - e0);
                en2 = dec_at(D[d] + SP[ep].off, SP[ep].len, cp, at);
                for (x = 0; x < en2 && pn < 4096u; ) {
                    unsigned cl = engram_cp_class(cp[x]);
                    size_t s0 = x;
                    if (cl == ENGRAM_CLASS_IDEO) {
                        while (x < en2 && x - s0 < 2u && engram_cp_class(cp[x]) == ENGRAM_CLASS_IDEO) x++;
                        if (x - s0 == 2u) { ps[pn] = s0; pl[pn] = 2u; pn++; }
                        continue;
                    }
                    if (cl == ENGRAM_CLASS_WORD) {
                        while (x < en2 && engram_cp_class(cp[x]) == ENGRAM_CLASS_WORD) x++;
                        if (x - s0 >= 3u) { ps[pn] = s0; pl[pn] = x - s0; pn++; }
                        continue;
                    }
                    x++;
                }
                if (pn < 2u) continue;
                hi = pn;
                if (a == 3u) {
                    size_t sp = (pn * 15u + 99u) / 100u;
                    if (sp < 2u) sp = 2u;
                    if (sp > pn) sp = pn;
                    lo = pn > sp ? (size_t)engram_rng_below(&rg, pn - sp + 1u) : 0u;
                    hi = lo + sp; want = sp;
                } else want = a == 0u ? 3u : a == 1u ? 5u : 8u;
                for (x = lo; x < hi; x++) perm[x - lo] = x;
                for (x = hi - lo; x > 1u; x--) {
                    size_t z = (size_t)engram_rng_below(&rg, x), tmp = perm[x - 1u];
                    perm[x - 1u] = perm[z]; perm[z] = tmp;
                }
                if (want > hi - lo) want = hi - lo;
                for (x = 0; x < want; x++) {
                    if (m) out[m++] = ' ';
                    for (y = 0; y < pl[perm[x]]; y++) out[m++] = cp[ps[perm[x]] + y];
                }
                tl = enc_cps(out, m, buf);
                cn[a]++;
                if (engram_store_recall(s, buf, tl, h, C, &nh) != ENGRAM_OK || !nh) continue;
                qc = cue_cps(&cfg, buf, tl);
                badhits += hits_ok(h, nh, ENGRAM_RANK_FULL, qc) + aligned_rule_ok(h, nh, qc) + (nh != C);
                rerank(h, nh, f);
                for (r = 0; r < 3u; r++) {
                    span_t e = SP[f[r] - 1u];
                    int all = e.doc == d;
                    for (x = 0; x < want && all; x++) {
                        size_t bs = SP[ep].off + at[ps[perm[x]]], be = SP[ep].off + at[ps[perm[x]] + pl[perm[x]]];
                        if (bs < e.off || be > e.off + e.len) all = 0;
                    }
                    ok[r] = all;
                    cr[a][r] += (unsigned)all;
                }
                for (k = 0; k < nh; k++) if (h[k].id - 1u == ep) { cand[1]++; break; }
                win[1] += ok[0] && !ok[1]; loss[1] += !ok[0] && ok[1];
            }
        }
    }
    ms = ((double)engram_now_ns() - t0) / 1e6;
    {
        unsigned nf = 0, nc = 0, sf[3] = { 0, 0, 0 }, sc[3] = { 0, 0, 0 };
        double pf[3], pc[3], sef, sec;
        for (a = 0; a < SC_FRAG; a++) {
            printf("       frag %2luB %lu typos  n=%3u   FULL %.3f  BAG %.3f  EXACT %.3f\n", (unsigned long)LENS[a % SC_LENS],
                   (unsigned long)(a / SC_LENS), fn[a], (double)fr[a][0] / fn[a], (double)fr[a][1] / fn[a], (double)fr[a][2] / fn[a]);
            for (r = 0; r < 3u; r++) sf[r] += fr[a][r];
            nf += fn[a];
        }
        for (a = 0; a < SC_CTRL; a++) {
            printf("       ctrl %-4s        n=%3u   FULL %.3f  BAG %.3f  EXACT %.3f\n", CTRL[a], cn[a],
                   (double)cr[a][0] / cn[a], (double)cr[a][1] / cn[a], (double)cr[a][2] / cn[a]);
            for (r = 0; r < 3u; r++) sc[r] += cr[a][r];
            nc += cn[a];
        }
        for (r = 0; r < 3u; r++) { pf[r] = (double)sf[r] / nf; pc[r] = (double)sc[r] / nc; }
        sef = sqrt(pf[0] * (1.0 - pf[0]) / nf);
        sec = sqrt(pc[0] * (1.0 - pc[0]) / nc);
        printf("       FRAGMENTS  FULL %.4f (se %.4f)  BAG %.4f  EXACT %.4f   candidates held the answer %.4f\n",
               pf[0], sef, pf[1], pf[2], (double)cand[0] / nf);
        printf("       CONTROL    FULL %.4f (se %.4f)  BAG %.4f  EXACT %.4f   candidates held the answer %.4f\n",
               pc[0], sec, pc[1], pc[2], (double)cand[1] / nc);
        printf("       FULL vs BAG, paired: fragments +%u -%u, control +%u -%u | %.2f ms per recall at %lu episodes\n",
               win[0], loss[0], win[1], loss[1], ms / (double)(nf + nc), (unsigned long)nsp);
        printf("       shuffled-target control: %u of %u (the criterion finds nothing where nothing is)\n", shuffled, shuffled_n);
        printf("RECALLPRINT %016llX\n", (unsigned long long)rp);

        ET_SECTION("S8: every answer obeys the ranking contract; the lesion stores agree with the re-ranking");
        ET_EQ_U64(badhits, 0u);
        ET_CHECK(lesion_n >= 20u);
        ET_EQ_U64(lesion_bad, 0u);
        ET_SECTION("S8: the instrument can fail -- an unrelated paragraph is (almost) never 'found'");
        ET_CHECK(shuffled * 100u <= shuffled_n);
        ET_SECTION("S8: FULL beats BAG beats EXACT on fragments; the alignment never loses a fragment");
        /* a sign test on the discordant pairs, at three standard deviations */
        ET_CHECKF((double)win[0] - (double)loss[0] >= 3.0 * sqrt((double)(win[0] + loss[0])) && win[0] > 0u,
                  "FULL vs BAG +%u -%u", win[0], loss[0]);
        ET_CHECK(loss[0] * 50u <= win[0]);
        ET_CHECKF(pf[1] - pf[2] >= 6.0 * sef, "BAG %.4f vs EXACT %.4f", pf[1], pf[2]);
        ET_SECTION("S8: the control -- the alignment costs keyword cues at most 1% net (measured: -2 of 1,197 dev,"
                   " +1 of 1,200 test)");
        ET_CHECKF(loss[1] <= win[1] + (nc + 99u) / 100u, "control: +%u -%u of %u", win[1], loss[1], nc);
        ET_SECTION("S8: floors (measured minus two standard errors)");
        ET_CHECKF(pf[0] >= (quick ? FLOOR_FRAG_Q : FLOOR_FRAG), "fragments FULL %.4f", pf[0]);
        ET_CHECKF(pc[0] >= (quick ? FLOOR_CTRL_Q : FLOOR_CTRL), "control FULL %.4f", pc[0]);
        ET_CHECKF((double)cand[0] / nf >= 0.95, "stage-1 candidates held the fragment's answer %.4f", (double)cand[0] / nf);
    }
out:
    engram_store_close(lesion[0]);
    engram_store_close(lesion[1]);
    engram_store_close(s);
    engram_free(h); engram_free(hl); engram_free(SP); engram_free(D); engram_free(DL); engram_free(raw);
}

int main(void)
{
    printf("ENGRAM P1.3 -- episodic store\n");
    ET_SELFTEST();
    test_contract();
    test_atomic();
    test_lifecycle();
    test_determinism_and_recall();
    test_ranking();
    {
        const char *q = getenv("ENGRAM_TEST_QUICK");
        test_scale(q && q[0] == '1');
    }
    return et_report("test_store");
}
