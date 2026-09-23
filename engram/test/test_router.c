/* ==================================================================================================
 * test_router.c -- P1.4: the router. Every clause of engram_router.h, checked.
 * ==================================================================================================
 *   R1  contract: refusals, empty router, duplicate ids, non-unit and non-finite vectors
 *   R2  EXACT is exact: equal to an independent double-precision reference; before training, and
 *       whenever nprobe covers every bucket, SEARCH returns the same hits as EXACT, bit for bit
 *   R3  every key routes to its own bucket: its own vector finds it first at nprobe 1
 *   R4  a random life of 20,000 operations against a plain model: after EVERY operation the internal
 *       invariants hold, every hit is live and correctly scored, and full-probe search equals the
 *       model's exhaustive answer
 *   R5  ALL OR NOTHING: every allocation of add (through map and bucket growth) and of train failed
 *       in turn -- E_MEM, fingerprint unchanged, invariants hold, and the retry reaches the clean state
 *   R6  the fingerprint is a function of the SET: two routers assembled in different orders, trained
 *       alike, agree; the print for the gate
 *   R7  AT SCALE: the dense vectors of every scale-corpus episode; recall of the exhaustive answer
 *       against keys actually scored, for near-copy and fragment cues; floors; ROUTEPRINT
 *   R8  GROWS WITHOUT A REBUILD: trained on half the keys, the other half added -- recall and balance
 *       against a router trained on all of them; the cost of an insert
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_router.h"
#include "../src/engram_chunk.h"
#include "../src/engram_enc.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "../src/engram_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

static int g_quick;

/* R7 floors: measured minus two standard errors, rounded down (C = 161, nprobe 8, 26,139 keys).
 *   full  (600 cues each)  near copies: the exhaustive best found 0.9983   fragments: top 10 0.6262
 *   quick (150 cues each)  near copies: 1.0000                             fragments: top 10 0.6247
 * Deterministic (ROUTEPRINT is compared across platforms), so a value under its floor means the code
 * changed. The fragment number is low BY THE NATURE OF THE SPACE, not by a defect -- research/p14_router. */
#define FLOOR_NEAR (g_quick ? 0.98 : 0.99)
#define FLOOR_FRAG (g_quick ? 0.54 : 0.58)

/* a random unit vector: Gaussian-ish components (sum of uniforms), normalised in double */
static void rand_unit(engram_rng *r, float *v, unsigned d)
{
    double s = 0.0;
    unsigned i;
    for (i = 0; i < d; i++) {
        double x = 0.0;
        unsigned k;
        for (k = 0; k < 4u; k++) x += (double)(engram_rng_u64(r) >> 11) * (1.0 / 9007199254740992.0) - 0.5;
        v[i] = (float)x;
        s += x * x;
    }
    s = sqrt(s);
    for (i = 0; i < d; i++) v[i] = (float)((double)v[i] / s);
}

/* clustered keys: a few hundred directions, each key a noisy copy -- a store with structure */
static void rand_clustered(engram_rng *r, const float *centres, unsigned nc, float *v, unsigned d, double noise)
{
    const float *c = centres + (size_t)engram_rng_below(r, nc) * d;
    double s = 0.0;
    unsigned i;
    rand_unit(r, v, d);
    for (i = 0; i < d; i++) { double x = (double)c[i] + noise * (double)v[i]; v[i] = (float)x; s += x * x; }
    s = sqrt(s);
    for (i = 0; i < d; i++) v[i] = (float)((double)v[i] / s);
}

static double ddot(const float *a, const float *b, unsigned d)
{
    double s = 0.0;
    unsigned i;
    for (i = 0; i < d; i++) s += (double)a[i] * (double)b[i];
    return s;
}

static engram_router *open_dim(unsigned dim, unsigned nprobe)
{
    engram_router_cfg c;
    engram_router *r = NULL;
    engram_router_cfg_default(&c);
    c.dim = dim;
    c.nprobe = nprobe;
    if (engram_router_open(&r, &c) != ENGRAM_OK) return NULL;
    return r;
}

/* ---- R1 ---------------------------------------------------------------------------------------- */
static void test_contract(void)
{
    engram_router *r = NULL;
    engram_router_cfg c;
    engram_route_hit h[8];
    engram_route_cost cost;
    size_t nh = 9;
    float v[64], w[64];
    engram_rng g;
    unsigned i;

    ET_SECTION("R1: refusals at open");
    ET_RC(engram_router_open(NULL, NULL), ENGRAM_E_ARG);
    engram_router_cfg_default(&c); c.dim = 0;   ET_RC(engram_router_open(&r, &c), ENGRAM_E_ARG);
    engram_router_cfg_default(&c); c.dim = 12;  ET_RC(engram_router_open(&r, &c), ENGRAM_E_ARG);
    engram_router_cfg_default(&c); c.dim = 4104; ET_RC(engram_router_open(&r, &c), ENGRAM_E_ARG);
    engram_router_cfg_default(&c); c.nprobe = 0; ET_RC(engram_router_open(&r, &c), ENGRAM_E_ARG);
    engram_router_cfg_default(&c); c.iters = 0;  ET_RC(engram_router_open(&r, &c), ENGRAM_E_ARG);
    ET_CHECK(r == NULL);

    ET_SECTION("R1: an empty router, and every refusal of add / remove / get / train / search");
    r = open_dim(64u, 4u);
    if (!r) { ET_CHECK(0); return; }
    engram_rng_seed(&g, 0x5101u);
    rand_unit(&g, v, 64u);
    ET_OK(engram_router_search(r, v, 4u, 0u, h, &nh, &cost));
    ET_EQ_U64(nh, 0u);
    ET_OK(engram_router_exact(r, v, 4u, h, &nh, NULL));
    ET_EQ_U64(nh, 0u);
    ET_RC(engram_router_train(r, 1u), ENGRAM_E_EMPTY);
    ET_RC(engram_router_add(r, 0u, v), ENGRAM_E_ARG);                  /* id 0 */
    ET_RC(engram_router_add(r, 1u, NULL), ENGRAM_E_ARG);
    for (i = 0; i < 64u; i++) w[i] = 2.0f * v[i];                      /* norm 2 */
    ET_RC(engram_router_add(r, 1u, w), ENGRAM_E_ARG);
    memcpy(w, v, sizeof w); w[5] = (float)NAN;
    ET_RC(engram_router_add(r, 1u, w), ENGRAM_E_ARG);
    memcpy(w, v, sizeof w); w[5] = (float)INFINITY;
    ET_RC(engram_router_add(r, 1u, w), ENGRAM_E_ARG);
    ET_OK(engram_router_add(r, 1u, v));
    ET_RC(engram_router_add(r, 1u, v), ENGRAM_E_EXISTS);
    ET_RC(engram_router_remove(r, 2u), ENGRAM_E_NOTFOUND);
    ET_RC(engram_router_remove(r, 0u), ENGRAM_E_ARG);
    ET_RC(engram_router_get(r, 2u, w), ENGRAM_E_NOTFOUND);
    ET_OK(engram_router_get(r, 1u, w));
    ET_CHECK(memcmp(v, w, sizeof v) == 0);
    ET_RC(engram_router_train(r, 0u), ENGRAM_E_ARG);
    ET_RC(engram_router_train(r, 2u), ENGRAM_E_ARG);                   /* more centroids than keys */
    ET_RC(engram_router_search(r, NULL, 4u, 0u, h, &nh, NULL), ENGRAM_E_ARG);
    ET_RC(engram_router_search(r, v, 0u, 0u, h, &nh, NULL), ENGRAM_E_ARG);
    ET_RC(engram_router_search(r, v, 4u, 0u, NULL, &nh, NULL), ENGRAM_E_ARG);
    for (i = 0; i < 64u; i++) w[i] = 0.5f * v[i];
    ET_RC(engram_router_search(r, w, 4u, 0u, h, &nh, NULL), ENGRAM_E_ARG);
    ET_OK(engram_router_search(r, v, 4u, 0u, h, &nh, &cost));
    ET_CHECK(nh == 1u && h[0].id == 1u && fabs((double)h[0].score - 1.0) < 1e-5);
    ET_EQ_U64(cost.keys_scored, 1u);
    ET_SECTION("R1: equal scores are ordered by id, and the cut at k keeps the lower ids");
    ET_OK(engram_router_add(r, 9u, v));                               /* the same vector under 1, 9, 3, 5 */
    ET_OK(engram_router_add(r, 3u, v));
    ET_OK(engram_router_add(r, 5u, v));
    ET_OK(engram_router_exact(r, v, 3u, h, &nh, NULL));
    ET_CHECK(nh == 3u && h[0].id == 1u && h[1].id == 3u && h[2].id == 5u);
    ET_OK(engram_router_search(r, v, 8u, 0u, h, &nh, NULL));
    ET_CHECK(nh == 4u && h[0].id == 1u && h[1].id == 3u && h[2].id == 5u && h[3].id == 9u);
    ET_OK(engram_router_remove(r, 9u)); ET_OK(engram_router_remove(r, 3u)); ET_OK(engram_router_remove(r, 5u));
    ET_OK(engram_router_train(r, 1u));
    ET_OK(engram_router_remove(r, 1u));
    ET_OK(engram_router_check(r));
    ET_OK(engram_router_search(r, v, 4u, 0u, h, &nh, &cost));
    ET_EQ_U64(nh, 0u);
    engram_router_close(r);
    engram_router_close(NULL);
}

/* ---- R2, R3 ------------------------------------------------------------------------------------ */
static void test_exact(void)
{
    enum { D = 64, N = 3000, NC = 40, Q = 200, K = 10 };
    static float keys[N * D], cen[NC * D], q[D];
    engram_router *r = open_dim(D, 4u);
    engram_route_hit a[K], b[K];
    engram_route_cost ca, cb;
    engram_rng g;
    size_t na, nb, i, j;
    unsigned bad = 0, bad_bits = 0, self_bad = 0, qi;
    if (!r) { ET_CHECK(0); return; }
    engram_rng_seed(&g, 0xE4AC7u);
    for (i = 0; i < NC; i++) rand_unit(&g, cen + i * D, D);
    for (i = 0; i < N; i++) {
        rand_clustered(&g, cen, NC, keys + i * D, D, 0.6);
        ET_OK(engram_router_add(r, 1000u + i * 7u, keys + i * D));
    }

    ET_SECTION("R2: EXACT equals an independent double-precision reference (top 10, 200 queries)");
    for (qi = 0; qi < Q; qi++) {
        double sc[N];
        rand_clustered(&g, cen, NC, q, D, 0.9);
        ET_OK(engram_router_exact(r, q, K, a, &na, &ca));
        ET_EQ_U64(ca.keys_scored, N);
        for (i = 0; i < N; i++) sc[i] = ddot(q, keys + i * D, D);
        for (j = 0; j < na; j++) {                 /* each hit's score, and nothing outside scores higher */
            size_t ki = (size_t)(a[j].id - 1000u) / 7u;
            if (fabs((double)a[j].score - sc[ki]) > 1e-5) bad++;
            if (j > 0u && (a[j].score > a[j - 1u].score ||
                           (a[j].score == a[j - 1u].score && a[j].id < a[j - 1u].id))) bad++;
        }
        for (i = 0; i < N; i++) {
            int in = 0;
            for (j = 0; j < na; j++) if (a[j].id == 1000u + i * 7u) in = 1;
            if (!in && sc[i] > (double)a[na - 1u].score + 1e-5) bad++;
        }
        /* untrained: the search IS the exact scan, to the bit */
        ET_OK(engram_router_search(r, q, K, 3u, b, &nb, &cb));
        if (na != nb || memcmp(a, b, na * sizeof *a) != 0) bad_bits++;
    }
    ET_EQ_U64(bad, 0u);
    ET_SECTION("R2: before training, SEARCH is EXACT to the bit");
    ET_EQ_U64(bad_bits, 0u);

    ET_SECTION("R2: after training, nprobe = C is EXACT to the bit; nprobe < C scores fewer keys");
    ET_OK(engram_router_train(r, 55u));
    ET_OK(engram_router_check(r));
    bad_bits = 0;
    {
        size_t fewer = 0;
        for (qi = 0; qi < Q; qi++) {
            rand_clustered(&g, cen, NC, q, D, 0.9);
            ET_OK(engram_router_exact(r, q, K, a, &na, NULL));
            ET_OK(engram_router_search(r, q, K, 55u, b, &nb, &cb));
            if (na != nb || memcmp(a, b, na * sizeof *a) != 0 || cb.keys_scored != N) bad_bits++;
            ET_OK(engram_router_search(r, q, K, 4u, b, &nb, &cb));
            if (cb.keys_scored < N && cb.buckets_probed == 4u && cb.centroids_scored == 55u) fewer++;
        }
        ET_EQ_U64(bad_bits, 0u);
        ET_EQ_U64(fewer, Q);
    }

    ET_SECTION("R3: every key finds itself first at nprobe 1 (it lives in its nearest centroid's bucket)");
    for (i = 0; i < N; i++) {
        ET_OK(engram_router_search(r, keys + i * D, 1u, 1u, a, &na, &ca));
        if (na != 1u || a[0].id != 1000u + i * 7u) self_bad++;
    }
    ET_EQ_U64(self_bad, 0u);
    engram_router_close(r);
}

/* ---- R4: a random life against a model ---------------------------------------------------------- */
static void test_life(void)
{
    enum { D = 32, MAXK = 1200 };
    static float mk[MAXK * D], v[D];
    static uint64_t mid[MAXK];
    size_t mn = 0;
    engram_router *r = open_dim(D, 3u);
    engram_route_hit h[16], e[16];
    engram_rng g;
    unsigned op, ops = g_quick ? 4000u : 20000u, inv_bad = 0, hit_bad = 0, full_bad = 0, trains = 0;
    unsigned bad_add = 0, bad_rm = 0, bad_train = 0, bad_count = 0;
    uint64_t next_id = 1;
    if (!r) { ET_CHECK(0); return; }
    engram_rng_seed(&g, 0x11FEu);
    ET_SECTION("R4: a random life -- adds, removes, retrains, searches -- against a plain model");
    for (op = 0; op < ops; op++) {
        unsigned kind = (unsigned)engram_rng_below(&g, 100u);
        if (kind < 45u && mn < MAXK) {                                  /* add (sometimes a duplicate id) */
            int dup = mn && engram_rng_below(&g, 10u) == 0u;          /* an id the model holds */
            uint64_t id = dup ? mid[engram_rng_below(&g, mn)] : next_id++;
            engram_rc rc;
            rand_unit(&g, v, D);
            rc = engram_router_add(r, id, v);
            if (dup) { if (rc != ENGRAM_E_EXISTS) bad_add++; }
            else if (rc != ENGRAM_OK) bad_add++;
            else { memcpy(mk + mn * D, v, sizeof v); mid[mn++] = id; }
        } else if (kind < 75u) {                                       /* remove (sometimes absent) */
            if (mn && engram_rng_below(&g, 8u) != 0u) {
                size_t i = (size_t)engram_rng_below(&g, mn);
                if (engram_router_remove(r, mid[i]) != ENGRAM_OK) bad_rm++;
                memcpy(mk + i * D, mk + (mn - 1u) * D, sizeof v); mid[i] = mid[mn - 1u]; mn--;
            } else if (engram_router_remove(r, next_id + 5u) != ENGRAM_E_NOTFOUND) bad_rm++;
        } else if (kind < 78u) {                                       /* retrain */
            if (mn) {
                unsigned C = 1u + (unsigned)engram_rng_below(&g, mn < 40u ? mn : 40u);
                if (engram_router_train(r, C) != ENGRAM_OK) bad_train++;
                trains++;
            } else if (engram_router_train(r, 1u) != ENGRAM_E_EMPTY) bad_train++;
        } else {                                                       /* search */
            size_t nh = 0, ne = 0, j, i;
            engram_router_stats st;
            rand_unit(&g, v, D);
            if (engram_router_search(r, v, 16u, 0u, h, &nh, NULL) != ENGRAM_OK) { hit_bad++; continue; }
            for (j = 0; j < nh; j++) {                                 /* every hit live, correctly scored */
                int live = 0;
                for (i = 0; i < mn; i++)
                    if (mid[i] == h[j].id) { live = 1; if (fabs(ddot(v, mk + i * D, D) - (double)h[j].score) > 1e-5) hit_bad++; }
                if (!live) hit_bad++;
            }
            engram_router_stats_get(r, &st);
            if (engram_router_search(r, v, 16u, (unsigned)st.buckets, h, &nh, NULL) != ENGRAM_OK ||
                engram_router_exact(r, v, 16u, e, &ne, NULL) != ENGRAM_OK || nh != ne ||
                memcmp(h, e, nh * sizeof *h) != 0 || ne != (mn < 16u ? mn : 16u)) full_bad++;
        }
        if (engram_router_check(r) != ENGRAM_OK) inv_bad++;
        {
            engram_router_stats st;
            engram_router_stats_get(r, &st);
            if (st.keys != mn) bad_count++;
        }
    }
    printf("       %u operations, %u retrains, %lu keys at the end\n", ops, trains, (unsigned long)mn);
    ET_EQ_U64(inv_bad, 0u);
    ET_EQ_U64(bad_add, 0u);
    ET_EQ_U64(bad_rm, 0u);
    ET_EQ_U64(bad_train, 0u);
    ET_EQ_U64(bad_count, 0u);
    ET_EQ_U64(hit_bad, 0u);
    ET_EQ_U64(full_bad, 0u);
    engram_router_close(r);
}

/* ---- R5: all or nothing ------------------------------------------------------------------------- */
typedef engram_rc (*router_op)(engram_router *r, const float *v);

static engram_rc op_add(engram_router *r, const float *v) { return engram_router_add(r, 999999u, v); }
static engram_rc op_train(engram_router *r, const float *v) { (void)v; return engram_router_train(r, 12u); }

static void sweep(const char *what, size_t prefix_keys, int prefix_train, router_op op)
{
    enum { D = 32 };
    static float v[D];
    engram_router *r;
    engram_rng g;
    size_t i, pk, m, bad_rc = 0, bad_fp = 0, bad_retry = 0;
    engram_router_stats s0, s1;
    uint64_t calls, f0, fok;
    engram_rc rc;
#define BUILD()                                                                                 \
    do {                                                                                       \
        r = open_dim(D, 3u);                                                                   \
        if (!r) { ET_CHECK(0); return; }                                                       \
        engram_rng_seed(&g, 0x5EEDu);                                                          \
        for (pk = 0; pk < prefix_keys; pk++) { rand_unit(&g, v, D); (void)engram_router_add(r, pk + 1u, v); } \
        if (prefix_train) (void)engram_router_train(r, 8u);                                    \
        rand_unit(&g, v, D);                                                                   \
    } while (0)
    for (;;) {                        /* the smallest prefix >= the one asked for whose op must allocate */
        BUILD();
        calls = engram_alloc_calls();
        ET_OK(op(r, v));
        m = (size_t)(engram_alloc_calls() - calls);
        fok = engram_router_fingerprint(r);
        engram_router_close(r);
        if (m > 0u || prefix_keys > 4096u) break;
        prefix_keys++;
    }
    for (i = 1; i <= m; i++) {
        BUILD();
        f0 = engram_router_fingerprint(r);
        engram_router_stats_get(r, &s0);
        engram_alloc_fail_at(i);
        rc = op(r, v);
        engram_alloc_fail_at(0);
        engram_router_stats_get(r, &s1);
        if (rc != ENGRAM_E_MEM) bad_rc++;
        if (engram_router_fingerprint(r) != f0 || engram_router_check(r) != ENGRAM_OK) bad_fp++;
        if (s1.keys != s0.keys || s1.buckets != s0.buckets || s1.trained != s0.trained || s1.adds != s0.adds ||
            s1.removes != s0.removes || s1.trains != s0.trains) bad_fp++;           /* the counters too */
        if (op(r, v) != ENGRAM_OK || engram_router_fingerprint(r) != fok || engram_router_check(r) != ENGRAM_OK) bad_retry++;
        engram_router_close(r);
    }
#undef BUILD
    printf("       %-40s %lu allocation points, each failed in turn (prefix %lu keys)\n", what, (unsigned long)m,
           (unsigned long)prefix_keys);
    ET_CHECKF(m > 0u, "%s: no allocation point -- the sweep would prove nothing", what);
    ET_CHECKF(bad_rc == 0u, "%s: %lu failures did not return E_MEM", what, (unsigned long)bad_rc);
    ET_CHECKF(bad_fp == 0u, "%s: %lu failures CHANGED the router", what, (unsigned long)bad_fp);
    ET_CHECKF(bad_retry == 0u, "%s: %lu retries did not reach the clean state", what, (unsigned long)bad_retry);
}

static void test_atomic(void)
{
    ET_SECTION("R5: all or nothing -- every allocation failed in turn");
    sweep("add to an empty router", 0u, 0, op_add);
    sweep("add that rehashes the id map (8 -> 9 keys)", 8u, 0, op_add);
    sweep("add into a trained router", 40u, 1, op_add);
    sweep("train 12 centroids over 200 keys", 200u, 0, op_train);
    sweep("retrain 12 centroids over 200 trained keys", 200u, 1, op_train);
    {   /* search scratch */
        engram_router *r = open_dim(32u, 3u);
        engram_route_hit h[64];
        size_t nh = 5;
        float v[32];
        engram_rng g;
        uint64_t f0;
        unsigned i;
        ET_CHECK(r != NULL);
        if (!r) return;
        engram_rng_seed(&g, 0x5C4u);
        for (i = 0; i < 100u; i++) { rand_unit(&g, v, 32u); ET_OK(engram_router_add(r, i + 1u, v)); }
        ET_OK(engram_router_train(r, 10u));
        f0 = engram_router_fingerprint(r);
        engram_alloc_fail_at(1);
        ET_RC(engram_router_search(r, v, 64u, 3u, h, &nh, NULL), ENGRAM_E_MEM);
        engram_alloc_fail_at(0);
        ET_EQ_U64(nh, 0u);
        ET_EQ_U64(engram_router_fingerprint(r), f0);
        ET_OK(engram_router_search(r, v, 64u, 3u, h, &nh, NULL));
        engram_router_close(r);
    }
}

/* ---- R6: a function of the set ------------------------------------------------------------------- */
static void test_set(void)
{
    enum { D = 32, N = 500 };
    static float keys[N * D];
    engram_router *a = open_dim(D, 3u), *b = open_dim(D, 3u);
    engram_rng g;
    size_t i;
    ET_SECTION("R6: two routers assembled in different orders, with removes, train to the same fingerprint");
    if (!a || !b) { ET_CHECK(0); engram_router_close(a); engram_router_close(b); return; }
    engram_rng_seed(&g, 0x5E7u);
    for (i = 0; i < N; i++) rand_unit(&g, keys + i * D, D);
    for (i = 0; i < N; i++) ET_OK(engram_router_add(a, i + 1u, keys + i * D));
    for (i = N; i-- > 0u; ) ET_OK(engram_router_add(b, i + 1u, keys + i * D));
    ET_OK(engram_router_add(b, 777777u, keys));                      /* b takes a detour */
    ET_OK(engram_router_remove(b, 777777u));
    for (i = 0; i < N; i += 5u) { ET_OK(engram_router_remove(a, i + 1u)); ET_OK(engram_router_remove(b, i + 1u)); }
    ET_EQ_U64(engram_router_fingerprint(a), engram_router_fingerprint(b));
    ET_OK(engram_router_train(a, 20u));
    ET_OK(engram_router_train(b, 20u));
    ET_EQ_U64(engram_router_fingerprint(a), engram_router_fingerprint(b));
    ET_OK(engram_router_remove(b, 2u));
    ET_CHECK(engram_router_fingerprint(a) != engram_router_fingerprint(b));
    engram_router_close(a);
    engram_router_close(b);
}

/* ---- R7, R8: at scale ------------------------------------------------------------------------------ */
/* every episode of the scale corpus, cut as the store cuts it, as a dense unit vector */
static float *scale_keys(size_t *n_out, char ***texts, size_t **lens)
{
    char *path;
    uint8_t *raw = NULL;
    size_t rawn = 0, i, n = 0, cap = 0;
    float *keys = NULL;
    engram_enc_cfg ec;
    *n_out = 0;
    path = engram_path_join(data_dir(), "scale_docs.txt");
    if (!path || engram_file_read(path, &raw, &rawn) != ENGRAM_OK) { engram_free(path); return NULL; }
    engram_free(path);
    engram_enc_cfg_default(&ec);
    *texts = NULL; *lens = NULL;
    for (i = 0; i < rawn; ) {
        size_t e = i, off, len;
        engram_chunkit it;
        while (e < rawn && raw[e] != '\n') e++;
        if (e > i && engram_chunkit_init(&it, raw + i, e - i, ENGRAM_EPI_TAIL, 0u) == ENGRAM_OK)
            while (engram_chunkit_next(&it, &off, &len)) {
                size_t c1 = cap, c2 = cap, c3 = cap;
                if (n + 1u > cap) {
                    if (engram_grow((void **)&keys, &c1, (n + 1u) * ENGRAM_D, sizeof *keys) != ENGRAM_OK ||
                        engram_grow((void **)texts, &c2, n + 1u, sizeof **texts) != ENGRAM_OK ||
                        engram_grow((void **)lens, &c3, n + 1u, sizeof **lens) != ENGRAM_OK) { engram_free(raw); return NULL; }
                    cap = c2 < c3 ? c2 : c3;
                    if (c1 / ENGRAM_D < cap) cap = c1 / ENGRAM_D;
                }
                if (engram_encode(&ec, raw + i + off, len, keys + n * ENGRAM_D, NULL) != ENGRAM_OK) continue;
                (*texts)[n] = (char *)engram_malloc(len);
                if (!(*texts)[n]) continue;
                memcpy((*texts)[n], raw + i + off, len);
                (*lens)[n] = len;
                n++;
            }
        i = e + 1u;
    }
    engram_free(raw);
    *n_out = n;
    return keys;
}

static void test_scale(void)
{
    enum { K = 10, NP = 6 };
    static const unsigned PROBES[NP] = { 1u, 2u, 4u, 8u, 16u, 32u };
    size_t n = 0, i, j, nq;
    char **texts = NULL;
    size_t *lens = NULL;
    float *keys, q[ENGRAM_D];
    engram_router *r = NULL, *half = NULL;
    engram_router_cfg rc;
    engram_route_hit ex[K], ap[K];
    engram_route_cost cost;
    engram_enc_cfg ec;
    engram_rng g;
    unsigned C, p, pass;
    double t0, t_train;
    uint64_t rp = 0x524F555445ull;
    double rec[2][NP], scored[2][NP], rec_half[NP], scored_half[NP], top1[2][NP], dq;

    ET_SECTION("R7: at scale -- every scale-corpus episode as a key");
    keys = scale_keys(&n, &texts, &lens);
    if (!keys || n < 20000u) { ET_CHECKF(0, "scale_docs.txt did not load"); goto out; }
    engram_router_cfg_default(&rc);
    rc.train_max = g_quick ? 8192u : 65536u;
    ET_OK(engram_router_open(&r, &rc));
    ET_OK(engram_router_open(&half, &rc));
    if (!r || !half) goto out;
    for (i = 0; i < n; i++) ET_OK(engram_router_add(r, i + 1u, keys + i * ENGRAM_D));
    C = (unsigned)sqrt((double)n);
    t0 = (double)engram_now_ns();
    ET_OK(engram_router_train(r, C));
    t_train = ((double)engram_now_ns() - t0) / 1e9;
    {
        engram_router_stats st;
        engram_router_stats_get(r, &st);
        printf("       %lu keys, C = %u trained in %.1f s (k-means on %lu): largest bucket %lu, %lu empty,"
               " imbalance %.2f\n", (unsigned long)n, C, t_train, (unsigned long)(n < rc.train_max ? n : rc.train_max),
               (unsigned long)st.largest_bucket, (unsigned long)st.empty_buckets, st.imbalance);
        ET_CHECK(st.imbalance < 3.0);
        ET_OK(engram_router_check(r));
    }
    /* R8's router: trained on the first half, then the second half added without retraining */
    for (i = 0; i < n / 2u; i++) ET_OK(engram_router_add(half, i + 1u, keys + i * ENGRAM_D));
    ET_OK(engram_router_train(half, C));
    t0 = (double)engram_now_ns();
    for (i = n / 2u; i < n; i++) ET_OK(engram_router_add(half, i + 1u, keys + i * ENGRAM_D));
    {
        double us = ((double)engram_now_ns() - t0) / 1e3 / (double)(n - n / 2u);
        engram_router_stats st;
        engram_router_stats_get(half, &st);
        printf("       R8: %lu keys added after training at %.1f us each (C = %u dot products and an append);"
               " imbalance %.2f\n", (unsigned long)(n - n / 2u), us, C, st.imbalance);
        ET_OK(engram_router_check(half));
    }

    engram_enc_cfg_default(&ec);
    memset(rec, 0, sizeof rec); memset(scored, 0, sizeof scored); memset(top1, 0, sizeof top1);
    memset(rec_half, 0, sizeof rec_half); memset(scored_half, 0, sizeof scored_half);
    nq = g_quick ? 150u : 600u;
    dq = (double)nq;
    engram_rng_seed(&g, 0xC0E5u);
    for (pass = 0; pass < 2u; pass++)              /* 0: near copies (2 typos)  1: fragments (a fifth) */
        for (j = 0; j < nq; j++) {
            size_t src = (size_t)((j * 7919u + 13u) % n), qlen = lens[src], a0 = 0;
            static char buf[ENGRAM_EPI_TAIL + 8];
            size_t na, nb;
            memcpy(buf, texts[src], qlen);
            if (pass == 0) {                       /* two substitutions of ASCII letters */
                unsigned t;
                for (t = 0; t < 2u; t++) {
                    size_t at = (size_t)engram_rng_below(&g, qlen);
                    if (buf[at] >= 'a' && buf[at] <= 'z') buf[at] = (char)('a' + (buf[at] - 'a' + 7) % 26);
                }
            } else {                               /* a fifth, from a third of the way in, at a byte boundary
                                                      that begins a codepoint */
                a0 = qlen / 3u;
                while (a0 < qlen && ((unsigned char)buf[a0] & 0xC0u) == 0x80u) a0++;
                qlen = (qlen - a0) / 5u + 8u;
                if (a0 + qlen > lens[src]) qlen = lens[src] - a0;
                while (a0 + qlen < lens[src] && ((unsigned char)buf[a0 + qlen] & 0xC0u) == 0x80u) qlen++;
            }
            if (engram_encode(&ec, buf + a0, qlen, q, NULL) != ENGRAM_OK) continue;
            ET_OK(engram_router_exact(r, q, K, ex, &na, NULL));
            for (p = 0; p < NP; p++) {
                size_t got = 0, x, y;
                ET_OK(engram_router_search(r, q, K, PROBES[p], ap, &nb, &cost));
                for (x = 0; x < na; x++) for (y = 0; y < nb; y++) if (ex[x].id == ap[y].id) { got++; break; }
                rec[pass][p] += (double)got / (double)na;
                top1[pass][p] += (nb > 0u && ap[0].id == ex[0].id) ? 1.0 : 0.0;
                scored[pass][p] += (double)(cost.keys_scored + cost.centroids_scored) / (double)n;
                if (p == 3u) for (y = 0; y < nb; y++) { rp = engram_mix2(rp, ap[y].id); rp = engram_mix2(rp, engram_hash_bytes(&ap[y].score, sizeof ap[y].score, 0u)); }
                if (pass == 1u) {
                    ET_OK(engram_router_search(half, q, K, PROBES[p], ap, &nb, &cost));
                    got = 0;
                    for (x = 0; x < na; x++) for (y = 0; y < nb; y++) if (ex[x].id == ap[y].id) { got++; break; }
                    rec_half[p] += (double)got / (double)na;
                    scored_half[p] += (double)(cost.keys_scored + cost.centroids_scored) / (double)n;
                }
            }
        }
    for (pass = 0; pass < 2u; pass++) {
        printf("       %s cues:\n", pass ? "fragment" : "near-copy");
        for (p = 0; p < NP; p++)
            printf("         nprobe %2u  finds the exhaustive best %.4f  and top 10 %.4f  scoring %.4f of the keys%s\n",
                   PROBES[p], top1[pass][p] / dq, rec[pass][p] / dq, scored[pass][p] / dq, p == 3u ? "   (default)" : "");
    }
    printf("       R8 (trained on half, the rest added): fragment cues\n");
    for (p = 0; p < NP; p++)
        printf("         nprobe %2u  recall %.4f (trained on all: %.4f)  scoring %.4f of the keys\n", PROBES[p],
               rec_half[p] / dq, rec[1][p] / dq, scored_half[p] / dq);
    printf("ROUTEPRINT %016llX\n", (unsigned long long)engram_mix2(rp, engram_router_fingerprint(r)));
    ET_SECTION("R7: floors -- at the default nprobe the router finds the exhaustive answer while scoring a"
               " small fraction of the keys");
    ET_CHECKF(top1[0][3] / dq >= FLOOR_NEAR, "near-copy: the exhaustive best found %.4f", top1[0][3] / dq);
    ET_CHECKF(rec[1][3] / dq >= FLOOR_FRAG, "fragment recall %.4f", rec[1][3] / dq);
    ET_CHECKF(scored[1][3] / dq <= 0.10, "scored %.4f of the keys", scored[1][3] / dq);
    for (p = 1; p < NP; p++) ET_CHECK(rec[1][p] >= rec[1][p - 1u] - 1e-12);    /* more probes never lose */
    ET_SECTION("R8: grown without a rebuild, recall stays within 0.03 of a router trained on everything");
    ET_CHECKF(rec_half[3] / dq >= rec[1][3] / dq - 0.03, "half-trained %.4f vs %.4f", rec_half[3] / dq, rec[1][3] / dq);
out:
    engram_router_close(r);
    engram_router_close(half);
    for (i = 0; texts && i < n; i++) engram_free(texts[i]);
    engram_free(texts); engram_free(lens); engram_free(keys);
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = q && q[0] == '1';
    printf("ENGRAM P1.4 -- router\n");
    ET_SELFTEST();
    test_contract();
    test_exact();
    test_life();
    test_atomic();
    test_set();
    test_scale();
    return et_report("test_router");
}
