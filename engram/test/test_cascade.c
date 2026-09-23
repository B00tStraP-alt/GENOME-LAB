/* ==================================================================================================
 * test_cascade.c -- P2.1.2: the Cascade weight and the hashed contexts, proven.
 * ==================================================================================================
 *   C1  the word: value and depth at every position, for K = 4, 16, 64
 *   C2  initialisation: equal to the independent reference; uniform over [0, 3K); refusals; every
 *       allocation failed in turn
 *   C3  the update against an independent Python implementation of the header (cascade_vectors.h):
 *       positions, moved and flipped counts, six steps, eight shapes -- bit for bit
 *   C4  the update's promises, measured: a zero gradient moves nothing; the expected move IS the
 *       desired move (unbiased); at most one basin a step; moves toward zero slowed by exactly
 *       2^(-kappa d 4/K), moves away never; kappa 0 slows nothing; positions never leave [0, 3K)
 *   C5  partition invariance: rows split any way, and the transposed update over features, give the
 *       same bytes as one pass (the property deterministic threading rests on)
 *   C6  the integer kernels against naive sums; gather refuses an id outside the matrix
 *   C7  persistence: round trip; every doctored field refused; every truncation; allocation sweep
 *   X1  contexts: refusals; order k lands in block k; ids depend on exactly the k bytes before the
 *       cursor; "before the start" is its own symbol; buckets filled evenly over a corpus
 *   C8  CASCADEPRINT and CTXPRINT -- compared across platforms by the gate
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_buf.h"
#include "../src/engram_cascade.h"
#include "../src/engram_ctx.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "cascade_vectors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_quick;

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

/* ---- C1 ------------------------------------------------------------------------------------------ */
static void test_word(void)
{
    static const uint32_t Ks[3] = { 4u, 16u, 64u };
    unsigned i, bad = 0;
    ET_SECTION("C1: the word -- value and depth at every position, K = 4, 16, 64");
    for (i = 0; i < 3u; i++) {
        uint32_t K = Ks[i], p;
        for (p = 0; p < 3u * K; p++) {
            int v = engram_cascade_value(p, K);
            uint32_t d = engram_cascade_depth(p, K);
            int want_v = p < K ? -1 : (p < 2u * K ? 0 : 1);
            uint32_t want_d = p < K ? K - 1u - p : (p < 2u * K ? p - K : p - 2u * K);
            if (v != want_v || d != want_d || d >= K) bad++;
        }
        /* the flip boundaries: depth 0 on either side of each boundary */
        if (engram_cascade_depth(K - 1u, K) != 0u || engram_cascade_depth(2u * K, K) != 0u) bad++;
    }
    ET_EQ_U64(bad, 0u);
}

/* ---- C2, C3 -------------------------------------------------------------------------------------- */
static void test_init_and_reference(void)
{
    size_t c;
    unsigned s;
    engram_cascade m;
    ET_SECTION("C2: initialisation -- equal to the reference; uniform over [0, 3K); refusals");
    ET_RC(engram_cascade_init(NULL, 1, 1, 16, 0.5f, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 0, 5, 16, 0.5f, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 5, 0, 16, 0.5f, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 5, 5, 8, 0.5f, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 5, 5, 16, -0.1f, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 5, 5, 16, (float)NAN, 1u), ENGRAM_E_ARG);
    ET_RC(engram_cascade_init(&m, 5, 5, 16, 17.0f, 1u), ENGRAM_E_ARG);
    ET_CHECK(m.pos == NULL && m.val == NULL);
    {   /* uniform: chi-square over the 48 positions of 480,000 draws, df 47: mean 47, sd 9.7 */
        double chi = 0.0, e;
        size_t cnt[48], i;
        ET_OK(engram_cascade_init(&m, 480u, 1000u, 16u, 0.5f, 77u));
        memset(cnt, 0, sizeof cnt);
        for (i = 0; i < 480000u; i++) cnt[m.pos[i]]++;
        e = 480000.0 / 48.0;
        for (i = 0; i < 48u; i++) chi += ((double)cnt[i] - e) * ((double)cnt[i] - e) / e;
        ET_CHECKF(chi < 47.0 + 4.0 * 9.7, "chi-square %.1f", chi);
        for (i = 0; i < 480000u; i++) if (m.val[i] != (int8_t)engram_cascade_value(m.pos[i], 16u)) { ET_CHECK(0); break; }
        engram_cascade_free(&m);
        ET_CHECK(m.pos == NULL);
        printf("       480,000 initial positions: chi-square %.1f (df 47)\n", chi);
    }
    {   /* allocation failure: E_MEM, nothing held */
        uint64_t k;
        for (k = 1; k <= 2u; k++) {
            engram_alloc_reset_run();
            engram_alloc_fail_at(k);
            ET_RC(engram_cascade_init(&m, 10u, 10u, 16u, 0.5f, 1u), ENGRAM_E_MEM);
            engram_alloc_fail_at(0u);
            ET_CHECK(m.pos == NULL && m.val == NULL);
        }
    }

    ET_SECTION("C3: the update against the independent reference -- 8 shapes x 6 steps, bit for bit");
    for (c = 0; c < CASCADE_CASES_N; c++) {
        const cascade_case *v = &cascade_cases[c];
        size_t n = (size_t)v->rows * v->cols;
        if (engram_cascade_init(&m, v->rows, v->cols, v->K, v->kappa, v->seed) != ENGRAM_OK) { ET_CHECK(0); continue; }
        ET_CHECKF(memcmp(m.pos, v->init, n) == 0, "case %zu: initial positions", c);
        for (s = 0; s < v->steps; s++) {
            uint64_t mv = 0, fl = 0;
            engram_cascade_update_rows(&m, v->grads + s * n, v->lr, (uint64_t)s + 1u, 0u, v->rows, &mv, &fl);
            engram_cascade_commit_step(&m, (uint64_t)s + 1u);
            ET_CHECKF(memcmp(m.pos, v->after + s * n, n) == 0, "case %zu step %u: positions", c, s + 1u);
            ET_CHECKF(mv == v->moved[s] && fl == v->flipped[s], "case %zu step %u: moved %llu/%u flipped %llu/%u", c,
                      s + 1u, (unsigned long long)mv, v->moved[s], (unsigned long long)fl, v->flipped[s]);
        }
        ET_EQ_U64(m.step, v->steps);
        engram_cascade_free(&m);
    }
    printf("       %u cases, every position after every step equal to the reference\n", (unsigned)CASCADE_CASES_N);
}

/* ---- C4 ------------------------------------------------------------------------------------------ */
/* One row of n identical weights at position p0, one update with a gradient that asks each for the
 * same move u: returns the mean number of positions moved (signed). */
static double mean_move(uint32_t K, float kappa, uint32_t p0, float g, float lr, uint32_t n, uint64_t step,
                        uint32_t *minp, uint32_t *maxp)
{
    engram_cascade m;
    float *gr;
    double sum = 0.0;
    uint32_t i;
    if (engram_cascade_init(&m, 1u, n, K, kappa, 0xC4u) != ENGRAM_OK) return -1e9;
    gr = (float *)engram_array(n, sizeof *gr);
    if (!gr) { engram_cascade_free(&m); return -1e9; }
    for (i = 0; i < n; i++) { m.pos[i] = (uint8_t)p0; m.val[i] = (int8_t)engram_cascade_value(p0, K); gr[i] = g; }
    engram_cascade_update_rows(&m, gr, lr, step, 0u, 1u, NULL, NULL);
    *minp = 255u; *maxp = 0u;
    for (i = 0; i < n; i++) {
        sum += (double)m.pos[i] - (double)p0;
        if (m.pos[i] < *minp) *minp = m.pos[i];
        if (m.pos[i] > *maxp) *maxp = m.pos[i];
        if (m.val[i] != (int8_t)engram_cascade_value(m.pos[i], K)) sum += 1e9;      /* value plane out of step */
    }
    engram_free(gr);
    engram_cascade_free(&m);
    return sum / (double)n;
}

static void test_update_promises(void)
{
    const uint32_t N = 200000u;
    uint32_t lo, hi;
    double mv, want, se;
    engram_cascade m;
    ET_SECTION("C4: a zero gradient moves nothing -- not one weight, at any learning rate");
    {
        float *z = (float *)engram_calloc(64u * 64u, sizeof *z);
        uint64_t moved = 0, fp;
        ET_OK(engram_cascade_init(&m, 64u, 64u, 16u, 0.5f, 3u));
        fp = engram_cascade_fingerprint(&m);
        if (z) engram_cascade_update_rows(&m, z, 1000.0f, 1u, 0u, 64u, &moved, NULL);
        ET_EQ_U64(moved, 0u);
        ET_EQ_U64(engram_cascade_fingerprint(&m), fp);
        engram_free(z);
        engram_cascade_free(&m);
    }

    ET_SECTION("C4: unbiased -- the mean move equals the desired move (within 4 standard errors)");
    /* all weights equal: rms = |g|, so u = -lr * sign(g) exactly: the desired move is lr positions */
    {
        static const float lrs[4] = { 0.3f, 1.0f, 2.7f, 9.4f };
        unsigned i;
        for (i = 0; i < 4u; i++) {
            /* start at 0-basin centre-ish (p = 24 for K 16) moving UP: never toward zero, no slowing */
            mv = mean_move(16u, 0.5f, 20u, -1.0f, lrs[i], N, 7u, &lo, &hi);
            want = (double)lrs[i];
            se = 0.5 / sqrt((double)N);
            ET_CHECKF(fabs(mv - want) < 4.0 * se + 1e-9, "lr %.1f: mean move %.5f, want %.5f", (double)lrs[i], mv, want);
            ET_CHECKF(hi - lo <= 1u, "lr %.1f: moves span %u..%u -- floor and floor + 1 only", (double)lrs[i], lo, hi);
        }
        printf("       lr 0.3, 1.0, 2.7, 9.4: mean moves within 4 SE of the desired move, each exactly floor or floor+1\n");
    }

    ET_SECTION("C4: at most one basin a step, and positions never leave [0, 3K)");
    mv = mean_move(16u, 0.0f, 20u, -1.0f, 1000.0f, 1000u, 9u, &lo, &hi);
    ET_CHECK(fabs(mv - 16.0) < 1e-9 && lo == 36u && hi == 36u);
    mv = mean_move(16u, 0.0f, 40u, -1.0f, 1000.0f, 1000u, 9u, &lo, &hi);            /* 40 + 16 clamps to 47 */
    ET_CHECK(lo == 47u && hi == 47u);
    mv = mean_move(16u, 0.0f, 5u, 1.0f, 1000.0f, 1000u, 9u, &lo, &hi);              /* 5 - 16 clamps to 0 */
    ET_CHECK(lo == 0u && hi == 0u);
    mv = mean_move(64u, 0.0f, 190u, -1.0f, 3.0f, 1000u, 9u, &lo, &hi);              /* K 64: top is 191 */
    ET_CHECK(hi <= 191u);

    ET_SECTION("C4: metaplasticity -- toward zero slowed by exactly 2^(-kappa d 4/K), away never, kappa 0 never");
    {
        /* a +1 weight at depth d (p = 2K + d) pushed DOWN (toward zero) by lr 1: mean move = -2^(-kappa d 4/K) */
        static const uint32_t ds[4] = { 0u, 3u, 8u, 15u };
        unsigned i;
        for (i = 0; i < 4u; i++) {
            double w = pow(2.0, -0.5 * (double)ds[i] * 4.0 / 16.0);
            mv = mean_move(16u, 0.5f, 32u + ds[i], 1.0f, 1.0f, N, 11u, &lo, &hi);
            se = sqrt(w * (1.0 - w) / (double)N) + 1e-12;
            ET_CHECKF(fabs(mv + w) < 4.0 * se + 1e-6, "depth %u toward zero: %.5f, want %.5f", ds[i], mv, -w);
            mv = mean_move(16u, 0.5f, 32u + ds[i] < 47u ? 32u + ds[i] : 46u, -1.0f, 1.0f, N, 11u, &lo, &hi);
            ET_CHECKF(fabs(mv - 1.0) < 1e-9 || 32u + ds[i] >= 47u, "depth %u away from zero: %.5f, want 1", ds[i], mv);
            mv = mean_move(16u, 0.0f, 32u + ds[i], 1.0f, 1.0f, N, 11u, &lo, &hi);
            ET_CHECKF(fabs(mv + 1.0) < 1e-9, "kappa 0, depth %u: %.5f, want -1", ds[i], mv);
        }
        /* and the mirror: a -1 weight pushed UP */
        mv = mean_move(16u, 0.5f, 7u, -1.0f, 1.0f, N, 13u, &lo, &hi);                 /* depth 8 */
        want = pow(2.0, -0.5 * 8.0 * 4.0 / 16.0);
        ET_CHECKF(fabs(mv - want) < 4.0 * sqrt(want * (1 - want) / (double)N), "-1 at depth 8 pushed up: %.5f want %.5f", mv, want);
        printf("       depth 0, 3, 8, 15: toward-zero moves scaled by 2^(-kappa d/4) within 4 SE; away and kappa 0 unscaled\n");
    }
}

/* ---- C5 ------------------------------------------------------------------------------------------ */
static void test_partitions(void)
{
    enum { R = 37, C = 211, F = 300, H = 24, NF = 37 };
    engram_cascade a, b, t, u;
    engram_rng r;
    float *g = (float *)engram_array((size_t)R * C, sizeof(float));
    float *gt = (float *)engram_array((size_t)H * F, sizeof(float));
    float *gs = (float *)engram_array((size_t)NF * H, sizeof(float));
    uint32_t feat[NF];
    unsigned step, i, h, f, bad = 0, badt = 0;
    uint64_t m1 = 0, m2 = 0, f1 = 0, f2 = 0;
    ET_SECTION("C5: partition invariance -- rows split any way give the same bytes as one pass");
    if (!g || !gt || !gs) { ET_CHECK(0); goto out; }
    ET_OK(engram_cascade_init(&a, R, C, 16u, 0.5f, 5u));
    ET_OK(engram_cascade_init(&b, R, C, 16u, 0.5f, 5u));
    engram_rng_seed(&r, 17u);
    for (step = 1; step <= 25u; step++) {
        uint32_t cut1 = (uint32_t)engram_rng_below(&r, R + 1u), cut2;
        for (i = 0; i < (unsigned)(R * C); i++) g[i] = (float)engram_rng_normal(&r) * (step % 4u == 0 ? 1e-4f : 1.0f);
        cut2 = cut1 + (uint32_t)engram_rng_below(&r, R - cut1 + 1u);
        engram_cascade_update_rows(&a, g, 0.8f, step, 0u, R, &m1, &f1);
        engram_cascade_update_rows(&b, g, 0.8f, step, cut2, R, &m2, &f2);        /* out of order, uneven */
        engram_cascade_update_rows(&b, g, 0.8f, step, 0u, cut1, &m2, &f2);
        engram_cascade_update_rows(&b, g, 0.8f, step, cut1, cut2, &m2, &f2);
        if (memcmp(a.pos, b.pos, (size_t)R * C) != 0) bad++;
    }
    ET_EQ_U64(bad, 0u);
    ET_CHECK(m1 == m2 && f1 == f2 && m1 > 0u);
    engram_cascade_free(&a); engram_cascade_free(&b);

    ET_SECTION("C5: the transposed update over features == the row update of the transpose, split any way");
    ET_OK(engram_cascade_init(&t, F, H, 16u, 0.5f, 5u));                  /* stored F x H */
    ET_OK(engram_cascade_init(&u, H, F, 16u, 0.5f, 5u));                  /* logical H x F */
    for (f = 0; f < F; f++) for (h = 0; h < H; h++) { u.pos[h * F + f] = t.pos[f * H + h]; u.val[h * F + f] = t.val[f * H + h]; }
    m1 = m2 = f1 = f2 = 0;
    for (step = 1; step <= 25u; step++) {
        uint32_t cut = (uint32_t)engram_rng_below(&r, H + 1u);
        memset(gt, 0, (size_t)H * F * sizeof *gt);
        for (i = 0; i < NF; i++) feat[i] = i * 8u + (unsigned)engram_rng_below(&r, 8u);
        for (i = 0; i < NF; i++) for (h = 0; h < H; h++) {
            float x = (float)engram_rng_normal(&r);
            gs[i * H + h] = x; gt[h * F + feat[i]] = x;
        }
        engram_cascade_update_units(&t, feat, NF, gs, 1.5f, step, cut, H, &m1, &f1);
        engram_cascade_update_units(&t, feat, NF, gs, 1.5f, step, 0u, cut, &m1, &f1);
        engram_cascade_update_rows(&u, gt, 1.5f, step, 0u, H, &m2, &f2);
        for (f = 0; f < F; f++) for (h = 0; h < H; h++) if (u.pos[h * F + f] != t.pos[f * H + h]) badt++;
    }
    ET_EQ_U64(badt, 0u);
    ET_CHECK(m1 == m2 && f1 == f2 && m1 > 0u);
    printf("       25 updates each: 3-way uneven row splits and 2-way unit splits, identical bytes and counts\n");
    engram_cascade_free(&t); engram_cascade_free(&u);
out:
    engram_free(g); engram_free(gt); engram_free(gs);
}

/* ---- C6 ------------------------------------------------------------------------------------------ */
static void test_kernels(void)
{
    enum { R = 70, C = 333 };
    engram_cascade m;
    engram_rng r;
    int8_t x[C];
    int32_t y[R], y2[R], gy[C];
    float d[R], yt[C], yt2[C];
    uint32_t rows[9] = { 0u, 5u, 5u, 69u, 12u, 0u, 33u, 33u, 33u };
    unsigned i, c, bad = 0;
    ET_SECTION("C6: the integer kernels against naive sums; any row split; gather refuses an outside id");
    ET_OK(engram_cascade_init(&m, R, C, 16u, 0.5f, 21u));
    engram_rng_seed(&r, 3u);
    for (c = 0; c < C; c++) x[c] = (int8_t)((int)engram_rng_below(&r, 256u) - 128);
    for (i = 0; i < R; i++) d[i] = (float)engram_rng_normal(&r);
    d[3] = 0.0f;
    engram_cascade_matvec(&m, x, y, 0u, R);
    engram_cascade_matvec(&m, x, y2, 40u, R);
    engram_cascade_matvec(&m, x, y2, 0u, 40u);
    for (i = 0; i < R; i++) {
        int64_t s = 0;
        for (c = 0; c < C; c++) s += (int64_t)engram_cascade_value(m.pos[i * C + c], 16u) * x[c];
        if (s != y[i] || y[i] != y2[i]) bad++;
    }
    engram_cascade_matvec_t(&m, d, yt, 0u, C);
    engram_cascade_matvec_t(&m, d, yt2, 100u, C);
    engram_cascade_matvec_t(&m, d, yt2, 0u, 100u);
    for (c = 0; c < C; c++) {
        float s = 0.0f;
        for (i = 0; i < R; i++) {
            int v = engram_cascade_value(m.pos[i * C + c], 16u);
            if (d[i] == 0.0f) continue;
            if (v > 0) s += d[i]; else if (v < 0) s -= d[i];
        }
        if (s != yt[c] || memcmp(&yt[c], &yt2[c], sizeof(float)) != 0) bad++;
    }
    ET_OK(engram_cascade_gather(&m, rows, 9u, gy));
    for (c = 0; c < C; c++) {
        int32_t s = 0;
        for (i = 0; i < 9u; i++) s += engram_cascade_value(m.pos[rows[i] * C + c], 16u);
        if (s != gy[c]) bad++;
    }
    ET_EQ_U64(bad, 0u);
    gy[0] = 12345;
    rows[4] = R;                                             /* one id past the end */
    ET_RC(engram_cascade_gather(&m, rows, 9u, gy), ENGRAM_E_ARG);
    ET_CHECK(gy[0] == 12345);                                /* refused before touching y */
    ET_OK(engram_cascade_gather(&m, rows, 0u, gy));          /* an empty bag is all zeros */
    for (c = 0; c < C; c++) if (gy[c] != 0) { ET_CHECK(0); break; }
    ET_RC(engram_cascade_gather(NULL, rows, 1u, gy), ENGRAM_E_ARG);
    engram_cascade_free(&m);
}

/* ---- C7 ------------------------------------------------------------------------------------------ */
static void test_persistence(void)
{
    engram_cascade m, l;
    uint8_t *buf = NULL, *doc = NULL;
    size_t n, used = 0, i, bad = 0;
    float *g;
    ET_SECTION("C7: persistence -- round trip; every doctored field refused; every truncation");
    ET_OK(engram_cascade_init(&m, 13u, 29u, 16u, 0.5f, 99u));
    g = (float *)engram_array(13u * 29u, sizeof *g);
    if (g) {
        for (i = 0; i < 13u * 29u; i++) g[i] = (float)((int)(i % 7u) - 3);
        engram_cascade_update_rows(&m, g, 2.0f, 1u, 0u, 13u, NULL, NULL);
        engram_cascade_commit_step(&m, 1u);
        engram_free(g);
    }
    n = engram_cascade_size(&m);
    ET_EQ_U64(n, 36u + 13u * 29u);
    buf = (uint8_t *)engram_malloc(n + 1u);
    doc = (uint8_t *)engram_malloc(n + 1u);
    if (!buf || !doc) { ET_CHECK(0); goto out; }
    engram_cascade_write(&m, buf);
    ET_OK(engram_cascade_read(&l, buf, n, 1000u, 1000u, &used));
    ET_EQ_U64(used, n);
    ET_EQ_U64(engram_cascade_fingerprint(&l), engram_cascade_fingerprint(&m));
    ET_CHECK(memcmp(l.val, m.val, n - 36u) == 0);
    engram_cascade_free(&l);
    buf[n] = 0xAAu;                                           /* bytes after the matrix are not read */
    ET_OK(engram_cascade_read(&l, buf, n + 1u, 1000u, 1000u, &used));
    ET_EQ_U64(used, n);
    engram_cascade_free(&l);
    {
        struct { size_t off; unsigned w; uint64_t v; engram_rc want; const char *what; } P[] = {
            { 0, 4, 2u, ENGRAM_E_VERSION, "version 2" },
            { 4, 4, 0u, ENGRAM_E_FORMAT, "rows 0" },
            { 4, 4, 1001u, ENGRAM_E_FORMAT, "rows over the caller's bound" },
            { 8, 4, 0u, ENGRAM_E_FORMAT, "cols 0" },
            { 8, 4, 1001u, ENGRAM_E_FORMAT, "cols over the caller's bound" },
            { 12, 4, 8u, ENGRAM_E_FORMAT, "K 8" },
            { 16, 4, 0x7FC00000u, ENGRAM_E_FORMAT, "kappa NaN" },
            { 16, 4, 0xBF800000u, ENGRAM_E_FORMAT, "kappa -1" },
            { 16, 4, 0x41880000u, ENGRAM_E_FORMAT, "kappa 17" },
            { 4, 4, 14u, ENGRAM_E_FORMAT, "one row more than the bytes hold" },
            { 36, 1, 48u, ENGRAM_E_FORMAT, "a position of 3K" },
            { 36 + 100, 1, 255u, ENGRAM_E_FORMAT, "a position of 255" },
        };
        unsigned k;
        for (k = 0; k < sizeof P / sizeof P[0]; k++) {
            engram_rc rc;
            memcpy(doc, buf, n);
            if (P[k].w == 4) engram_le_put_u32(doc + P[k].off, (uint32_t)P[k].v); else doc[P[k].off] = (uint8_t)P[k].v;
            memset(&l, 0x5A, sizeof l);
            rc = engram_cascade_read(&l, doc, n, 1000u, 1000u, &used);
            ET_CHECKF(rc == P[k].want, "%s: got %s", P[k].what, engram_rcname(rc));
            ET_CHECKF(used == 0u, "%s: used %zu", P[k].what, used);
        }
        printf("       %u doctored fields refused, nothing held\n", (unsigned)(sizeof P / sizeof P[0]));
    }
    for (i = 0; i < n; i++) {
        if (engram_cascade_read(&l, buf, i, 1000u, 1000u, &used) == ENGRAM_OK) { bad++; engram_cascade_free(&l); }
    }
    ET_EQ_U64(bad, 0u);
    {
        uint64_t k;
        for (k = 1; k <= 2u; k++) {
            engram_alloc_reset_run();
            engram_alloc_fail_at(k);
            ET_RC(engram_cascade_read(&l, buf, n, 1000u, 1000u, &used), ENGRAM_E_MEM);
            engram_alloc_fail_at(0u);
            ET_EQ_U64(used, 0u);
        }
    }
out:
    engram_free(buf); engram_free(doc);
    engram_cascade_free(&m);
}

/* ---- X1 ------------------------------------------------------------------------------------------ */
typedef struct { uint64_t key; uint32_t id; } ctx_pair;

static int ctx_pair_cmp(const void *x, const void *y)
{
    const ctx_pair *a = (const ctx_pair *)x, *b = (const ctx_pair *)y;
    if (a->key != b->key) return a->key < b->key ? -1 : 1;
    return a->id < b->id ? -1 : a->id > b->id;
}

static void test_ctx(void)
{
    engram_ctx_cfg c;
    uint32_t a[ENGRAM_CTX_MAX_ORDERS], b[ENGRAM_CTX_MAX_ORDERS];
    const uint8_t *t = (const uint8_t *)"the cat sat on the mat, and the cat sat still";
    size_t n = strlen((const char *)t), i;
    unsigned k, bad = 0;
    ET_SECTION("X1: contexts -- refusals; the feature space");
    engram_ctx_cfg_default(&c);
    ET_OK(engram_ctx_cfg_check(&c));
    ET_EQ_U64(engram_ctx_features(&c), (uint64_t)c.orders << c.buckets_log2);
    c.orders = 0; ET_RC(engram_ctx_cfg_check(&c), ENGRAM_E_ARG);
    c.orders = ENGRAM_CTX_MAX_ORDERS + 1u; ET_RC(engram_ctx_cfg_check(&c), ENGRAM_E_ARG);
    engram_ctx_cfg_default(&c); c.buckets_log2 = 3u; ET_RC(engram_ctx_cfg_check(&c), ENGRAM_E_ARG);
    c.buckets_log2 = 21u; ET_RC(engram_ctx_cfg_check(&c), ENGRAM_E_ARG);
    ET_RC(engram_ctx_cfg_check(NULL), ENGRAM_E_ARG);

    ET_SECTION("X1: order k lands in block k; ids depend on EXACTLY the k bytes before the cursor");
    engram_ctx_cfg_default(&c);
    for (i = 0; i <= n; i++) {
        engram_ctx_ids(&c, t, n, i, a);
        for (k = 0; k < c.orders; k++) if (a[k] >> c.buckets_log2 != k) bad++;
    }
    ET_EQ_U64(bad, 0u);
    /* "the cat sat" occurs twice: at every cursor in the repeat, orders reaching back only into the
     * repeat agree, and the first order reaching outside it may not */
    {
        const char *second = strstr((const char *)t + 5, "cat sat");
        size_t p1 = 4u, p2 = second ? (size_t)(second - (const char *)t) : 0u;   /* "cat sat", twice */
        unsigned same = 0, want = 0;
        size_t off;
        ET_CHECK(p2 > p1);
        for (off = 1; off <= 7u; off++) {
            engram_ctx_ids(&c, t, n, p1 + off, a);
            engram_ctx_ids(&c, t, n, p2 + off, b);
            for (k = 0; k < c.orders; k++) {
                if (k + 1u <= off) { want++; same += a[k] == b[k]; }
            }
        }
        ET_EQ_U64(same, want);
    }
    {   /* change the byte 3 before the cursor: orders 1, 2 unchanged; orders 3.. change (collisions
         * possible in principle -- at 4096 buckets, none on this text) */
        uint8_t u[64];
        memcpy(u, t, n);
        u[20 - 3] ^= 0x20u;
        engram_ctx_ids(&c, t, n, 20u, a);
        engram_ctx_ids(&c, u, n, 20u, b);
        ET_CHECK(a[0] == b[0] && a[1] == b[1]);
        for (k = 2; k < c.orders; k++) ET_CHECK(a[k] != b[k]);
    }

    ET_SECTION("X1: before the start is its own symbol; t == n is valid; t > n clamps to n");
    {   /* order 1 at t = 0 hashes the symbol 256, "before the start", which is not a byte. At a million
         * buckets a collision among 257 symbols has probability ~3e-2 in total, so any clash here would
         * say the hash INPUT is the same -- that the marker is not its own symbol. (At the default 4096
         * buckets the 257 order-1 symbols collide by the birthday bound, ~8 pairs: that is bucketing, not
         * identity, and the higher orders tell them apart.) */
        engram_ctx_cfg wide = c;
        unsigned clash = 0;
        wide.buckets_log2 = 20u;
        engram_ctx_ids(&wide, t, n, 0u, a);
        for (k = 0; k < 256u; k++) {
            uint8_t one = (uint8_t)k;
            engram_ctx_ids(&wide, &one, 1u, 1u, b);
            if (a[0] == b[0]) clash++;
        }
        ET_EQ_U64(clash, 0u);
    }
    engram_ctx_ids(&c, t, n, n + 5u, a);
    engram_ctx_ids(&c, t, n, n, b);
    ET_CHECK(memcmp(a, b, c.orders * sizeof a[0]) == 0);

    ET_SECTION("X1: over a corpus the DISTINCT contexts of every order spread evenly over its buckets");
    {
        char *path = engram_path_join(data_dir(), "corpus_en.txt");
        uint8_t *raw = NULL;
        size_t rn = 0;
        ctx_pair *pr;
        uint32_t *cnt;
        if (!path || engram_file_read(path, &raw, &rn) != ENGRAM_OK) { ET_CHECKF(0, "corpus_en.txt did not load"); engram_free(path); return; }
        engram_free(path);
        pr = (ctx_pair *)engram_array(rn, sizeof *pr);
        cnt = (uint32_t *)engram_array((size_t)1 << c.buckets_log2, sizeof *cnt);
        if (pr && cnt) {
            size_t B = (size_t)1 << c.buckets_log2;
            for (k = 1; k < c.orders; k++) {           /* orders 2 .. 6: enough distinct contexts to fill */
                size_t np = 0, d = 0, j;
                double chi = 0.0, e, sd;
                for (i = k + 1u; i <= rn; i++) {     /* contexts that lie wholly inside the text */
                    engram_ctx_ids(&c, raw, rn, i, a);
                    pr[np].key = engram_hash_bytes(raw + i - (k + 1u), k + 1u, 0u);
                    pr[np].id = a[k];
                    np++;
                }
                qsort(pr, np, sizeof *pr, ctx_pair_cmp);
                memset(cnt, 0, B * sizeof *cnt);
                for (j = 0; j < np; j++) {
                    if (j && pr[j].key == pr[j - 1u].key) {
                        if (pr[j].id != pr[j - 1u].id) bad++;   /* one context, two ids: not a function */
                        continue;
                    }
                    cnt[pr[j].id & (B - 1u)]++;
                    d++;
                }
                e = (double)d / (double)B;
                for (j = 0; j < B; j++) chi += ((double)cnt[j] - e) * ((double)cnt[j] - e) / e;
                sd = sqrt(2.0 * (double)(B - 1u));
                ET_CHECKF(fabs(chi - (double)(B - 1u)) < 5.0 * sd, "order %u: %zu distinct contexts, chi-square %.0f (df %zu)",
                          k + 1u, d, chi, B - 1u);
                printf("       order %u: %zu distinct contexts over %zu buckets, chi-square %.0f (df %zu, sd %.0f)\n",
                       k + 1u, d, B, chi, B - 1u, sd);
            }
            ET_EQ_U64(bad, 0u);
        }
        engram_free(pr); engram_free(cnt); engram_free(raw);
    }
}

/* ---- C8 ------------------------------------------------------------------------------------------ */
static void test_prints(void)
{
    engram_cascade m;
    engram_ctx_cfg c;
    engram_rng r;
    float *g;
    uint32_t id[ENGRAM_CTX_MAX_ORDERS];
    uint64_t h = 0x43545850ull;
    const uint8_t *t = (const uint8_t *)"Het was een koude, heldere dag in april \xe2\x80\x94 \xe5\x8d\x81\xe4\xba\x8c.";
    size_t i, n = strlen((const char *)t);
    unsigned s;
    ET_SECTION("C8: CASCADEPRINT and CTXPRINT -- compared across platforms by the gate");
    ET_OK(engram_cascade_init(&m, 64u, 200u, 16u, 0.5f, 0xBEEFu));
    g = (float *)engram_array(64u * 200u, sizeof *g);
    engram_rng_seed(&r, 42u);
    for (s = 1; g && s <= 40u; s++) {
        for (i = 0; i < 64u * 200u; i++) g[i] = (float)engram_rng_normal(&r) * (float)(1 + s % 3);
        engram_cascade_update_rows(&m, g, 0.1f * (float)(s % 7 + 1), s, 0u, 64u, NULL, NULL);
        engram_cascade_commit_step(&m, s);
    }
    engram_free(g);
    printf("CASCADEPRINT %016llX\n", (unsigned long long)engram_cascade_fingerprint(&m));
    engram_cascade_free(&m);
    engram_ctx_cfg_default(&c);
    for (i = 0; i <= n; i++) {
        unsigned k;
        engram_ctx_ids(&c, t, n, i, id);
        for (k = 0; k < c.orders; k++) h = engram_mix2(h, id[k]);
    }
    printf("CTXPRINT %016llX\n", (unsigned long long)h);
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = q && q[0] == '1';
    printf("ENGRAM P2.1.2 -- the Cascade weight and the hashed contexts\n");
    ET_SELFTEST();
    test_word();
    test_init_and_reference();
    test_update_promises();
    test_partitions();
    test_kernels();
    test_persistence();
    test_ctx();
    test_prints();
    return et_report("test_cascade");
}
