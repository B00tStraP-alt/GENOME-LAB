/* ==================================================================================================
 * test_expert.c -- P2.1.3: the expert's forward pass, proven.
 * ==================================================================================================
 *   E1  the configuration: the default is P2.1.1's choice; every field's refusal; the size bound
 *   E2  the forward pass against an independent Python implementation (expert_vectors.h): the bits of
 *       every position and the logits of three, for five configurations -- the default among them --
 *       with non-trivial gains and biases, to the bit
 *   E3  the contract: range refusals, an empty range is 0 bits, t == n is a position
 *   E4  what must hold of any distribution: probabilities sum to 1; no position costs negative bits;
 *       all-zero logits cost exactly 8 bits a byte; bits over [a, c) = bits over [a, b) + [b, c)
 *   E5  every allocation of open failed in turn: E_MEM, nothing leaked
 *   E6  identity: the same cfg, the same expert; a changed seed or gain, a different fingerprint; size
 *   E7  speed, measured; EXPERTPRINT for the gate
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_expert.h"
#include "../src/engram_math.h"
#include "../src/engram_plat.h"
#include "expert_vectors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int g_quick;

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

/* the fixed gains and biases the reference used: exact binary fractions */
static void set_pattern(engram_expert *e)
{
    engram_expert_units u;
    size_t i;
    engram_expert_units_get(e, &u);
    for (i = 0; i < u.n_gain; i++) u.gain[i] = (float)(0.25 + (double)((i * 37u) % 23u + 1u) / 16.0);
    for (i = 0; i < u.n_bias; i++) u.bias[i] = (float)((double)((int)((i * 13u) % 11u) - 5) / 8.0);
}

static engram_expert *open_case(unsigned width, unsigned hidden, unsigned orders, unsigned blog2, unsigned K, uint64_t seed)
{
    engram_expert_cfg c;
    engram_expert *e = NULL;
    engram_expert_cfg_default(&c);
    c.width = width; c.hidden = hidden; c.ctx.orders = orders; c.ctx.buckets_log2 = blog2; c.K = K; c.seed = seed;
    if (engram_expert_open(&e, &c) != ENGRAM_OK) return NULL;
    return e;
}

static int same_bits(double a, double b) { return memcmp(&a, &b, sizeof a) == 0; }

/* ---- E1 ------------------------------------------------------------------------------------------ */
static void test_cfg(void)
{
    engram_expert_cfg c, d;
    engram_expert *e = NULL;
    ET_SECTION("E1: the default is P2.1.1's choice -- 6 x 4096 contexts, width 512, no hidden layer, Cascade-16");
    engram_expert_cfg_default(&c);
    ET_CHECK(c.ctx.orders == 6u && c.ctx.buckets_log2 == 12u && c.width == 512u && c.hidden == 0u && c.K == 16u);
    ET_CHECK(c.kappa == 0.5f && c.lr == 0.1f);
    ET_OK(engram_expert_cfg_check(&c));

    ET_SECTION("E1: every field refused out of range; the size bound is E_FULL; nothing opened");
#define REFUSE(field, value, want) do { d = c; d.field = value; ET_RC(engram_expert_cfg_check(&d), want); \
                                        ET_RC(engram_expert_open(&e, &d), want); ET_CHECK(e == NULL); } while (0)
    REFUSE(width, 8u, ENGRAM_E_ARG);
    REFUSE(width, 520u, ENGRAM_E_ARG);             /* not a multiple of 16 */
    REFUSE(width, 4112u, ENGRAM_E_ARG);
    REFUSE(hidden, ENGRAM_EXPERT_MAX_HIDDEN + 1u, ENGRAM_E_ARG);
    REFUSE(K, 8u, ENGRAM_E_ARG);
    REFUSE(kappa, -0.5f, ENGRAM_E_ARG);
    REFUSE(kappa, (float)NAN, ENGRAM_E_ARG);
    REFUSE(lr, 0.0f, ENGRAM_E_ARG);
    REFUSE(lr, (float)NAN, ENGRAM_E_ARG);
    REFUSE(lr_unit, 0.0f, ENGRAM_E_ARG);
    REFUSE(lr_unit, 2.0f, ENGRAM_E_ARG);
    REFUSE(ctx.orders, 0u, ENGRAM_E_ARG);
    REFUSE(ctx.buckets_log2, 21u, ENGRAM_E_ARG);
    d = c; d.ctx.orders = 16u; d.ctx.buckets_log2 = 20u; d.width = 4096u;        /* 68 G weights */
    ET_RC(engram_expert_cfg_check(&d), ENGRAM_E_FULL);
    ET_RC(engram_expert_open(&e, &d), ENGRAM_E_FULL);
    d = c; d.hidden = 4u; d.width = 4096u;                                 /* 101 M + 4 x 16.8 M + 1 M: inside */
    ET_OK(engram_expert_cfg_check(&d));
    ET_RC(engram_expert_cfg_check(NULL), ENGRAM_E_ARG);
    ET_RC(engram_expert_open(NULL, &c), ENGRAM_E_ARG);
#undef REFUSE
    ET_CHECK(engram_expert_cfg_get(NULL) == NULL);
}

/* ---- E2 ------------------------------------------------------------------------------------------ */
static void test_reference(void)
{
    size_t c, t;
    ET_SECTION("E2: every position's bits and three positions' logits equal the independent reference, to the bit");
    for (c = 0; c < EXPERT_CASES_N; c++) {
        const expert_case *v = &expert_cases[c];
        engram_expert *e = open_case(v->width, v->hidden, v->orders, v->blog2, v->K, v->seed);
        unsigned bad = 0, badl = 0, k;
        double total = 0.0, sum = 0.0;
        float lg[256];
        if (!e) { ET_CHECKF(0, "case %zu did not open", c); continue; }
        set_pattern(e);
        for (t = 0; t < EXPERT_TEXT_N; t++) {
            double b;
            if (engram_expert_bits(e, expert_text, EXPERT_TEXT_N, t, t + 1u, &b) != ENGRAM_OK || !same_bits(b, v->bits[t])) bad++;
            sum += v->bits[t];
        }
        for (k = 0; k < 3u; k++) {
            if (engram_expert_logits(e, expert_text, EXPERT_TEXT_N, v->lpos[k], lg) != ENGRAM_OK ||
                memcmp(lg, v->logits + k * 256u, sizeof lg) != 0) badl++;
        }
        ET_OK(engram_expert_bits(e, expert_text, EXPERT_TEXT_N, 0u, EXPERT_TEXT_N, &total));
        ET_CHECKF(bad == 0u && badl == 0u && same_bits(total, sum),
                  "case %zu (W %u, hidden %u, %u x 2^%u, K %u): %u positions and %u logit rows differ; total %.17g vs %.17g",
                  c, v->width, v->hidden, v->orders, v->blog2, v->K, bad, badl, total, sum);
        engram_expert_close(e);
    }
    printf("       %u configurations x %u positions: bits and logits identical to the reference\n",
           (unsigned)EXPERT_CASES_N, (unsigned)EXPERT_TEXT_N);
}

/* ---- E3, E4 -------------------------------------------------------------------------------------- */
static void test_contract(void)
{
    engram_expert *e = open_case(64u, 1u, 4u, 8u, 16u, 5u);
    const uint8_t *t = expert_text;
    size_t n = EXPERT_TEXT_N, i;
    double b = 1.0, b1, b2, b3;
    float lg[256];
    engram_expert_units u;
    unsigned neg = 0, badsum = 0;
    ET_SECTION("E3: range refusals; an empty range is 0 bits; t == n is a position");
    if (!e) { ET_CHECK(0); return; }
    set_pattern(e);
    ET_RC(engram_expert_bits(e, t, n, 5u, 4u, &b), ENGRAM_E_ARG);
    ET_CHECK(b == 0.0);
    ET_RC(engram_expert_bits(e, t, n, 0u, n + 1u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_bits(e, NULL, 3u, 0u, 1u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_bits(NULL, t, n, 0u, 1u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_bits(e, t, n, 0u, 1u, NULL), ENGRAM_E_ARG);
    ET_OK(engram_expert_bits(e, t, n, 7u, 7u, &b));
    ET_CHECK(b == 0.0);
    ET_OK(engram_expert_bits(e, NULL, 0u, 0u, 0u, &b));
    ET_OK(engram_expert_logits(e, t, n, n, lg));
    ET_RC(engram_expert_logits(e, t, n, n + 1u, lg), ENGRAM_E_ARG);
    ET_RC(engram_expert_logits(e, t, n, 0u, NULL), ENGRAM_E_ARG);

    ET_SECTION("E4: probabilities sum to 1; no negative bits; zero logits cost exactly 8 bits; bits add over ranges");
    for (i = 0; i <= n; i++) {
        double m, z = 0.0, ps = 0.0;
        unsigned v;
        ET_OK(engram_expert_logits(e, t, n, i, lg));
        m = (double)lg[0];
        for (v = 1; v < 256u; v++) if ((double)lg[v] > m) m = (double)lg[v];
        for (v = 0; v < 256u; v++) z += engram_exp((double)lg[v] - m);
        for (v = 0; v < 256u; v++) ps += engram_exp((double)lg[v] - m) / z;
        if (fabs(ps - 1.0) > 1e-12) badsum++;
        if (i < n) {
            ET_OK(engram_expert_bits(e, t, n, i, i + 1u, &b));
            if (!(b >= 0.0)) neg++;
        }
    }
    ET_EQ_U64(badsum, 0u);
    ET_EQ_U64(neg, 0u);
    ET_OK(engram_expert_bits(e, t, n, 3u, 60u, &b1));
    ET_OK(engram_expert_bits(e, t, n, 3u, 31u, &b2));
    ET_OK(engram_expert_bits(e, t, n, 31u, 60u, &b3));
    ET_CHECKF(fabs(b1 - (b2 + b3)) <= 1e-9 * b1, "%.17g vs %.17g", b1, b2 + b3);
    engram_expert_units_get(e, &u);
    for (i = (size_t)(1u + 1u) * 64u; i < u.n_gain; i++) u.gain[i] = 0.0f;   /* the head's gains */
    for (i = 0; i < u.n_bias; i++) u.bias[i] = 0.0f;
    ET_OK(engram_expert_bits(e, t, n, 0u, n, &b));
    ET_CHECKF(b == 8.0 * (double)n, "uniform: %.17g bits over %zu bytes", b, n);
    engram_expert_close(e);
}

/* ---- E5 ------------------------------------------------------------------------------------------ */
static void test_faults(void)
{
    engram_expert_cfg c;
    engram_expert *e = NULL;
    uint64_t calls, k;
    unsigned wrong = 0;
    ET_SECTION("E5: every allocation of open failed in turn -- E_MEM, nothing held, nothing leaked");
    engram_expert_cfg_default(&c);
    c.width = 32u; c.hidden = 2u; c.ctx.buckets_log2 = 6u;
    engram_alloc_reset_run();
    ET_OK(engram_expert_open(&e, &c));
    calls = engram_alloc_calls();
    engram_expert_close(e);
    ET_CHECK(calls >= 10u);
    for (k = 1; k <= calls; k++) {
        engram_rc rc;
        engram_alloc_reset_run();
        engram_alloc_fail_at(k);
        e = (engram_expert *)1;
        rc = engram_expert_open(&e, &c);
        engram_alloc_fail_at(0u);
        if (rc != ENGRAM_E_MEM || e != NULL) wrong++;
    }
    ET_EQ_U64(wrong, 0u);
    printf("       %llu allocations, each failed in turn: E_MEM every time\n", (unsigned long long)calls);
}

/* ---- E6 ------------------------------------------------------------------------------------------ */
static void test_identity(void)
{
    engram_expert *a = open_case(48u, 1u, 3u, 7u, 16u, 9u), *b = open_case(48u, 1u, 3u, 7u, 16u, 9u);
    engram_expert *c = open_case(48u, 1u, 3u, 7u, 16u, 10u);
    engram_expert *d = NULL;
    engram_expert_units u;
    engram_expert_stats st;
    engram_expert_cfg cfg;
    ET_SECTION("E6: the same cfg, the same expert; a changed seed or a changed gain, a different fingerprint");
    if (!a || !b || !c) { ET_CHECK(0); goto out; }
    ET_EQ_U64(engram_expert_fingerprint(a), engram_expert_fingerprint(b));
    ET_CHECK(engram_expert_fingerprint(a) != engram_expert_fingerprint(c));
    engram_expert_units_get(b, &u);
    u.gain[5] = 1.0000001f;
    ET_CHECK(engram_expert_fingerprint(a) != engram_expert_fingerprint(b));
    u.gain[5] = 1.0f;
    u.bias[200] = -0.0f;                                   /* a different bit pattern is a different state */
    ET_CHECK(engram_expert_fingerprint(a) != engram_expert_fingerprint(b));
    ET_EQ_U64(engram_expert_fingerprint(NULL), 0u);

    ET_SECTION("E6: size -- the default holds 12,713,984 weights and 1,024 floats");
    engram_expert_cfg_default(&cfg);
    ET_OK(engram_expert_open(&d, &cfg));
    engram_expert_stats_get(d, &st);
    ET_EQ_U64(st.weights, 6u * 4096u * 512u + 256u * 512u);
    ET_EQ_U64(st.units, 512u + 256u + 256u);
    ET_CHECK(st.bytes_resident >= 2u * st.weights && st.bytes_resident < 2u * st.weights + 65536u);
    printf("       default expert: %zu weights, %zu floats, %.1f MB resident\n", st.weights, st.units,
           (double)st.bytes_resident / 1e6);
out:
    engram_expert_close(a); engram_expert_close(b); engram_expert_close(c); engram_expert_close(d);
}

/* ---- E7 ------------------------------------------------------------------------------------------ */
static void test_speed_and_print(void)
{
    engram_expert *e = NULL;
    char *path;
    uint8_t *raw = NULL;
    size_t rn = 0, span;
    double b = 0.0;
    uint64_t t0, h;
    ET_SECTION("E7: speed of the default expert's forward pass; EXPERTPRINT for the gate");
    path = engram_path_join(data_dir(), "corpus_en.txt");
    if (!path || engram_file_read(path, &raw, &rn) != ENGRAM_OK) { ET_CHECKF(0, "corpus_en.txt did not load"); engram_free(path); return; }
    engram_free(path);
    ET_OK(engram_expert_open(&e, NULL));
    if (e) {
        span = g_quick ? 4000u : 40000u;
        if (span > rn) span = rn;
        t0 = engram_now_ns();
        ET_OK(engram_expert_bits(e, raw, rn, 0u, span, &b));
        printf("       untrained default expert: %.3f bits/byte over %zu bytes, %.1f us a byte\n", b / (double)span, span,
               (double)(engram_now_ns() - t0) / 1e3 / (double)span);
        ET_CHECK(b / (double)span > 7.0 && b / (double)span < 12.0);     /* untrained: near uniform */
        ET_OK(engram_expert_bits(e, raw, rn, 0u, 4000u < rn ? 4000u : rn, &b));
        h = engram_mix2(engram_expert_fingerprint(e), 0u);
        {
            uint64_t bb;
            memcpy(&bb, &b, sizeof bb);
            h = engram_mix2(h, bb);
        }
        printf("EXPERTPRINT %016llX\n", (unsigned long long)h);
        engram_expert_close(e);
    }
    engram_free(raw);
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = q && q[0] == '1';
    printf("ENGRAM P2.1.3 -- the expert's forward pass\n");
    ET_SELFTEST();
    test_cfg();
    test_reference();
    test_contract();
    test_faults();
    test_identity();
    test_speed_and_print();
    return et_report("test_expert");
}
