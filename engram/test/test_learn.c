/* ==================================================================================================
 * test_learn.c -- P2.2: the slow store learns. P2.2.1: the backward pass, proven.
 * ==================================================================================================
 *   G1  the backward pass against the Python implementation of engram_expert.h's specification
 *       (grad_vectors.h, whose generator first proves the FORMULAS against finite differences): the
 *       loss, and the gradient of every bias, gain, head weight, hidden weight and bag row, to the bit,
 *       for six configurations -- the default among them
 *   G2  the loss the backward pass returns IS engram_expert_bits over the same positions (R7)
 *   G3  finite differences on the REAL network: the bias and the head's gains reach the loss through no
 *       quantisation, so a central difference of the C's own bits must agree with the C's gradient --
 *       and the comparison is shown able to fail
 *   G4  exact algebra: a batch of one position twice is the gradient of that position once, to the bit;
 *       the bag's rows are exactly the batch's distinct features
 *   G5  the contract: refusals, gradients replaced not accumulated, no allocation in backward, every
 *       allocation of grad_open failed in turn
 *   G6  speed, measured; GRADPRINT for the gate
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_expert.h"
#include "../src/engram_math.h"
#include "../src/engram_plat.h"
#include "grad_vectors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* the fixed gains and biases the reference used: exact binary fractions (as test_expert.c); signed:
 * every fifth gain negative, as a trained gain may become */
static void set_pattern_s(engram_expert *e, int sign)
{
    engram_expert_units u;
    size_t i;
    engram_expert_units_get(e, &u);
    for (i = 0; i < u.n_gain; i++) {
        u.gain[i] = (float)(0.25 + (double)((i * 37u) % 23u + 1u) / 16.0);
        if (sign && i % 5u == 3u) u.gain[i] = -u.gain[i];
    }
    for (i = 0; i < u.n_bias; i++) u.bias[i] = (float)((double)((int)((i * 13u) % 11u) - 5) / 8.0);
}
static void set_pattern(engram_expert *e) { set_pattern_s(e, 1); }

static engram_expert *open_case(unsigned width, unsigned hidden, unsigned orders, unsigned blog2, unsigned K, uint64_t seed)
{
    engram_expert_cfg c;
    engram_expert *e = NULL;
    engram_expert_cfg_default(&c);
    c.width = width; c.hidden = hidden; c.ctx.orders = orders; c.ctx.buckets_log2 = blog2; c.K = K; c.seed = seed;
    if (engram_expert_open(&e, &c) != ENGRAM_OK) return NULL;
    return e;
}

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, sizeof u); return u; }
static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, sizeof u); return u; }

/* FNV-1a-64 over the floats' bytes, little-endian -- the generator's hash */
static uint64_t fnv_floats(uint64_t h, const float *x, size_t n)
{
    size_t i;
    unsigned k;
    for (i = 0; i < n; i++) {
        uint32_t w = fbits(x[i]);
        for (k = 0; k < 4u; k++) { h ^= (w >> (8u * k)) & 0xFFu; h *= 0x100000001B3ull; }
    }
    return h;
}
#define FNV0 0xCBF29CE484222325ull

/* index of the first float whose bits differ from want, or n */
static size_t first_diff(const float *got, const uint32_t *want, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) if (fbits(got[i]) != want[i]) return i;
    return n;
}

/* ---- G1 ------------------------------------------------------------------------------------------ */
static void test_reference(void)
{
    size_t c;
    unsigned long before = et_fail;
    unsigned ok = 0;
    ET_SECTION("G1: the loss and every gradient equal the reference implementation of the specification, to the bit");
    for (c = 0; c < GRAD_CASES_N; c++) {
        const grad_case *v = &grad_cases[c];
        engram_expert *e = open_case(v->width, v->hidden, v->orders, v->blog2, v->K, v->seed);
        engram_expert_grad *g = NULL;
        engram_expert_grad_view gv;
        size_t pos[16], b, W = v->width, n_gain = (size_t)(v->hidden + 1u) * W + 256u, d;
        double bits = 0.0;
        uint64_t hh = FNV0;
        unsigned j;
        if (!e || v->B > 16u || engram_expert_grad_open(&g, e, v->B) != ENGRAM_OK) {
            ET_CHECKF(0, "case %zu did not open", c);
            engram_expert_close(e);
            continue;
        }
        if (v->pattern) set_pattern_s(e, v->pattern == 2);
        for (b = 0; b < v->B; b++) pos[b] = v->pos[b];
        ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pos, v->B, &bits));
        engram_expert_grad_get(g, &gv);
        ET_CHECKF(dbits(bits) == v->bits, "case %zu: loss %.17g, reference bits 0x%016llX", c, bits,
                  (unsigned long long)v->bits);
        ET_CHECKF((d = first_diff(gv.bias, v->bias, 256u)) == 256u, "case %zu: bias gradient differs at %zu", c, d);
        ET_CHECK(gv.n_gain == n_gain);
        ET_CHECKF((d = first_diff(gv.gain, v->gain, n_gain)) == n_gain, "case %zu: gain gradient differs at %zu of %zu", c, d, n_gain);
        if (v->head) ET_CHECKF((d = first_diff(gv.head, v->head, 256u * W)) == 256u * W, "case %zu: head gradient differs at [%zu][%zu]", c, d / W, d % W);
        ET_CHECKF(fnv_floats(FNV0, gv.head, 256u * W) == v->head_h, "case %zu: the head gradient's hash differs", c);
        for (j = 0; j < v->hidden; j++) {
            hh = fnv_floats(hh, gv.hidden[j], W * W);
            if (v->hid) ET_CHECKF((d = first_diff(gv.hidden[j], v->hid + (size_t)j * W * W, W * W)) == W * W,
                                  "case %zu: hidden %u gradient differs at [%zu][%zu]", c, j, d / W, d % W);
        }
        ET_CHECKF(hh == v->hid_h, "case %zu: the hidden layers' gradient hash differs", c);
        for (j = v->hidden; j < ENGRAM_EXPERT_MAX_HIDDEN; j++) ET_CHECK(gv.hidden[j] == NULL);
        ET_CHECKF(gv.n_feat == v->n_feat && memcmp(gv.feat, v->feat, v->n_feat * sizeof *v->feat) == 0,
                  "case %zu: %zu features, reference %u", c, gv.n_feat, v->n_feat);
        if (gv.n_feat == v->n_feat) {
            if (v->bag) ET_CHECKF((d = first_diff(gv.bag, v->bag, v->n_feat * W)) == v->n_feat * W,
                                  "case %zu: bag gradient differs at feature %zu unit %zu", c, d / W, d % W);
            ET_CHECKF(fnv_floats(FNV0, gv.bag, v->n_feat * W) == v->bag_h, "case %zu: the bag gradient's hash differs", c);
        }
        engram_expert_grad_close(g);
        engram_expert_close(e);
        if (et_fail == before) ok++;
        before = et_fail;
    }
    printf("       %u of %u configurations (hidden 0, 1, 2; W 16 .. 512; K 4, 16, 64; the default): identical to the reference\n",
           ok, (unsigned)GRAD_CASES_N);
}

/* ---- G2 ------------------------------------------------------------------------------------------ */
static void test_loss_is_bits(void)
{
    engram_expert *e = open_case(48u, 2u, 4u, 6u, 16u, 31u);
    engram_expert_grad *g = NULL;
    size_t pos[24], b;
    double lb = 0.0, sum = 0.0;
    ET_SECTION("G2: the backward pass's loss is engram_expert_bits over the same positions, summed in order, to the bit");
    if (!e || engram_expert_grad_open(&g, e, 24u) != ENGRAM_OK) { ET_CHECK(0); goto out; }
    set_pattern(e);
    for (b = 0; b < 24u; b++) pos[b] = (b * 7u + 3u) % GRAD_TEXT_N;
    ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pos, 24u, &lb));
    for (b = 0; b < 24u; b++) {
        double x = 0.0;
        ET_OK(engram_expert_bits(e, grad_text, GRAD_TEXT_N, pos[b], pos[b] + 1u, &x));
        sum += x;
    }
    ET_CHECKF(dbits(lb) == dbits(sum), "backward %.17g, bits %.17g", lb, sum);
out:
    engram_expert_grad_close(g);
    engram_expert_close(e);
}

/* ---- G3 ------------------------------------------------------------------------------------------ */
/* the mean loss in nats over the batch, as the gradient is of: bits * ln 2 / B */
static double mean_nats(engram_expert *e, const size_t *pos, size_t B)
{
    size_t b;
    double s = 0.0;
    for (b = 0; b < B; b++) {
        double x = 0.0;
        (void)engram_expert_bits(e, grad_text, GRAD_TEXT_N, pos[b], pos[b] + 1u, &x);
        s += x;
    }
    return s * 0.69314718055994530942 / (double)B;
}

/* |fd - an| within 2e-3 (|fd| + |an|) + 2e-6: logits are floats, so the difference quotient of a
 * double loss over a 2^-7 step carries ~1e-7 of rounding -- the floor sits well above it, the relative
 * term well above the step's curvature error (measured below, and the check shown able to fail) */
static double fd_ratio(double fd, double an) { return fabs(fd - an) / (2e-3 * (fabs(fd) + fabs(an)) + 2e-6); }

static void test_finite_differences(void)
{
    static const unsigned shapes[3][2] = { { 32u, 0u }, { 48u, 1u }, { 16u, 2u } };
    unsigned s;
    double worst = 0.0, worst_wrong = 1e300;
    unsigned checked = 0, wrong_caught = 0, eligible = 0;
    ET_SECTION("G3: central differences of the C's own loss agree with its gradient for the bias and head gains");
    for (s = 0; s < 3u; s++) {
        engram_expert *e = open_case(shapes[s][0], shapes[s][1], 3u, 5u, 16u, 40u + s);
        engram_expert_grad *g = NULL;
        engram_expert_grad_view gv;
        engram_expert_units u;
        size_t pos[6] = { 4, 9, 13, 22, 41, 60 }, k;
        double bits;
        if (!e || engram_expert_grad_open(&g, e, 6u) != ENGRAM_OK) { ET_CHECK(0); engram_expert_close(e); continue; }
        set_pattern(e);
        ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pos, 6u, &bits));
        engram_expert_grad_get(g, &gv);
        engram_expert_units_get(e, &u);
        for (k = 0; k < 64u; k++) {
            /* the bias of 32 bytes (the batch's targets among them), then 32 head gains */
            size_t v = k < 32u ? (k < 6u ? grad_text[pos[k]] : (k * 41u) % 256u) : (k * 29u) % 256u;
            float *p = k < 32u ? &u.bias[v] : &u.gain[u.n_gain - 256u + v];
            float an = k < 32u ? gv.bias[v] : gv.gain[gv.n_gain - 256u + v];
            float old = *p, hi = old + 0.0078125f, lo = old - 0.0078125f;   /* exact: the values are multiples of 1/16 */
            double lp, lm, fd, r;
            *p = hi; lp = mean_nats(e, pos, 6u);
            *p = lo; lm = mean_nats(e, pos, 6u);
            *p = old;
            fd = (lp - lm) / ((double)hi - (double)lo);
            r = fd_ratio(fd, (double)an);
            if (r > worst) worst = r;
            checked++;
            /* R6: the same comparison against a gradient 3% off must fail wherever the gradient is not tiny */
            if (fabs((double)an) > 1e-3) {
                double rw = fd_ratio(fd, (double)an * 1.03);
                eligible++;
                if (rw > 1.0) wrong_caught++;
                if (rw < worst_wrong) worst_wrong = rw;
            }
        }
        engram_expert_grad_close(g);
        engram_expert_close(e);
    }
    ET_CHECKF(worst <= 1.0, "worst |fd - an| is %.3f of the tolerance", worst);
    ET_CHECKF(eligible >= 40u && wrong_caught == eligible, "a gradient 3%% wrong was caught %u of %u times; best ratio %.3f",
              wrong_caught, eligible, worst_wrong);
    printf("       %u parameters over hidden 0, 1, 2: worst %.3f of the tolerance; a 3%% error caught %u of %u times\n",
           checked, worst, wrong_caught, eligible);
}

/* ---- G4 ------------------------------------------------------------------------------------------ */
static int same_grads(const engram_expert_grad *a, const engram_expert_grad *b, size_t W, unsigned hidden)
{
    engram_expert_grad_view x, y;
    unsigned j;
    engram_expert_grad_get(a, &x);
    engram_expert_grad_get(b, &y);
    if (x.n_gain != y.n_gain || x.n_feat != y.n_feat) return 0;
    if (memcmp(x.bias, y.bias, 256u * sizeof(float)) || memcmp(x.gain, y.gain, x.n_gain * sizeof(float)) ||
        memcmp(x.head, y.head, 256u * W * sizeof(float)) || memcmp(x.feat, y.feat, x.n_feat * sizeof(uint32_t)) ||
        memcmp(x.bag, y.bag, x.n_feat * W * sizeof(float)))
        return 0;
    for (j = 0; j < hidden; j++) if (memcmp(x.hidden[j], y.hidden[j], W * W * sizeof(float))) return 0;
    return 1;
}

static int u32_cmp(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return x < y ? -1 : x > y;
}

static void test_algebra(void)
{
    engram_expert *e = open_case(32u, 1u, 5u, 4u, 16u, 51u);
    engram_expert_grad *g1 = NULL, *g2 = NULL;
    engram_expert_grad_view gv;
    const engram_expert_cfg *cfg;
    size_t one[1], two[2], t, many[20], i, nd = 0;
    uint32_t ids[20u * ENGRAM_CTX_MAX_ORDERS], distinct[20u * ENGRAM_CTX_MAX_ORDERS];
    double b1, b2;
    unsigned bad = 0;
    ET_SECTION("G4: one position twice is the gradient of it once, to the bit (every step is linear in 1/B)");
    if (!e || engram_expert_grad_open(&g1, e, 20u) != ENGRAM_OK || engram_expert_grad_open(&g2, e, 20u) != ENGRAM_OK) {
        ET_CHECK(0); goto out;
    }
    set_pattern(e);
    for (t = 0; t < GRAD_TEXT_N; t += 9u) {
        one[0] = two[0] = two[1] = t;
        if (engram_expert_backward(e, g1, grad_text, GRAD_TEXT_N, one, 1u, &b1) != ENGRAM_OK ||
            engram_expert_backward(e, g2, grad_text, GRAD_TEXT_N, two, 2u, &b2) != ENGRAM_OK ||
            !same_grads(g1, g2, 32u, 1u) || dbits(b2) != dbits(b1 + b1))
            bad++;
    }
    ET_EQ_U64(bad, 0u);

    ET_SECTION("G4: the bag's rows are exactly the batch's distinct features, ascending");
    cfg = engram_expert_cfg_get(e);
    for (i = 0; i < 20u; i++) many[i] = (i * 5u) % 23u;                 /* repeats, and shared contexts */
    ET_OK(engram_expert_backward(e, g1, grad_text, GRAD_TEXT_N, many, 20u, &b1));
    for (i = 0; i < 20u; i++) engram_ctx_ids(&cfg->ctx, grad_text, GRAD_TEXT_N, many[i], ids + i * cfg->ctx.orders);
    qsort(ids, 20u * cfg->ctx.orders, sizeof *ids, u32_cmp);
    for (i = 0; i < 20u * cfg->ctx.orders; i++) if (nd == 0u || distinct[nd - 1u] != ids[i]) distinct[nd++] = ids[i];
    engram_expert_grad_get(g1, &gv);
    ET_CHECKF(gv.n_feat == nd && memcmp(gv.feat, distinct, nd * sizeof *distinct) == 0, "%zu features, want %zu",
              gv.n_feat, nd);
    ET_CHECK(nd < 20u * cfg->ctx.orders);                                /* the case does share features */
out:
    engram_expert_grad_close(g1); engram_expert_grad_close(g2);
    engram_expert_close(e);
}

/* ---- G5 ------------------------------------------------------------------------------------------ */
static void test_contract(void)
{
    engram_expert *e = open_case(32u, 1u, 3u, 5u, 16u, 61u), *other = open_case(48u, 1u, 3u, 5u, 16u, 61u);
    engram_expert_grad *g = NULL, *g2 = NULL, *gx = NULL;
    engram_expert_grad_view gv;
    size_t pa[4] = { 1, 20, 21, 50 }, pb[3] = { 7, 7, 60 }, bad_pos[2] = { 3, GRAD_TEXT_N };
    double b = 1.0, ba;
    uint64_t calls, k;
    unsigned wrong = 0;
    ET_SECTION("G5: refusals -- NULLs, B == 0, B > batch_max, a position >= n, another shape; bits zeroed");
    if (!e || !other) { ET_CHECK(0); goto out; }
    set_pattern(e);
    ET_RC(engram_expert_grad_open(NULL, e, 4u), ENGRAM_E_ARG);
    g = (engram_expert_grad *)1;
    ET_RC(engram_expert_grad_open(&g, NULL, 4u), ENGRAM_E_ARG); ET_CHECK(g == NULL);
    g = (engram_expert_grad *)1;
    ET_RC(engram_expert_grad_open(&g, e, 0u), ENGRAM_E_ARG); ET_CHECK(g == NULL);
    g = (engram_expert_grad *)1;
    ET_RC(engram_expert_grad_open(&g, e, ENGRAM_EXPERT_MAX_BATCH + 1u), ENGRAM_E_ARG); ET_CHECK(g == NULL);
    ET_OK(engram_expert_grad_open(&g, e, 4u));
    ET_OK(engram_expert_grad_open(&gx, other, 4u));
    if (!g || !gx) { ET_CHECK(0); goto out; }
    ET_RC(engram_expert_backward(NULL, g, grad_text, GRAD_TEXT_N, pa, 4u, &b), ENGRAM_E_ARG); ET_CHECK(b == 0.0);
    b = 1.0;
    ET_RC(engram_expert_backward(e, NULL, grad_text, GRAD_TEXT_N, pa, 4u, &b), ENGRAM_E_ARG); ET_CHECK(b == 0.0);
    ET_RC(engram_expert_backward(e, g, NULL, GRAD_TEXT_N, pa, 4u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, NULL, 4u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pa, 4u, NULL), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pa, 0u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pa, 5u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, bad_pos, 2u, &b), ENGRAM_E_ARG);
    ET_RC(engram_expert_backward(e, gx, grad_text, GRAD_TEXT_N, pa, 4u, &b), ENGRAM_E_ARG);
    engram_expert_grad_get(NULL, &gv);
    ET_CHECK(gv.bias == NULL && gv.n_gain == 0u && gv.n_feat == 0u);
    engram_expert_grad_get(g, NULL);                                    /* no crash */

    ET_SECTION("G5: the gradients are REPLACED: A, then B, then A again is A, to the bit; a larger workspace agrees");
    ET_OK(engram_expert_grad_open(&g2, e, 64u));
    if (!g2) { ET_CHECK(0); goto out; }
    ET_OK(engram_expert_backward(e, g2, grad_text, GRAD_TEXT_N, pa, 4u, &ba));
    ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pb, 3u, &b));
    ET_CHECK(!same_grads(g, g2, 32u, 1u));
    ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pa, 4u, &b));
    ET_CHECK(same_grads(g, g2, 32u, 1u) && dbits(b) == dbits(ba));

    ET_SECTION("G5: backward allocates nothing; every allocation of grad_open failed in turn -- E_MEM, nothing leaked");
    engram_alloc_reset_run();
    ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pa, 4u, &b));
    ET_EQ_U64(engram_alloc_calls(), 0u);
    engram_alloc_reset_run();
    engram_expert_grad_close(g2); g2 = NULL;
    ET_OK(engram_expert_grad_open(&g2, e, 8u));
    calls = engram_alloc_calls();
    engram_expert_grad_close(g2); g2 = NULL;
    ET_CHECK(calls >= 20u);
    for (k = 1; k <= calls; k++) {
        engram_rc rc;
        engram_alloc_reset_run();
        engram_alloc_fail_at(k);
        g2 = (engram_expert_grad *)1;
        rc = engram_expert_grad_open(&g2, e, 8u);
        engram_alloc_fail_at(0u);
        if (rc != ENGRAM_E_MEM || g2 != NULL) { wrong++; if (rc == ENGRAM_OK) engram_expert_grad_close(g2); }
        g2 = NULL;
    }
    ET_EQ_U64(wrong, 0u);
    printf("       %llu allocations of grad_open, each failed in turn: E_MEM every time\n", (unsigned long long)calls);
out:
    engram_expert_grad_close(g); engram_expert_grad_close(g2); engram_expert_grad_close(gx);
    engram_expert_close(e); engram_expert_close(other);
}

/* ---- G6 ------------------------------------------------------------------------------------------ */
static void test_speed_and_print(void)
{
    engram_expert *e = NULL;
    engram_expert_grad *g = NULL;
    engram_expert_grad_view gv;
    size_t pos[64], b, B = 64u;
    double bits = 0.0;
    uint64_t t0, dt, h;
    unsigned j;
    ET_SECTION("G6: speed of the default expert's backward pass; GRADPRINT for the gate");
    ET_OK(engram_expert_open(&e, NULL));
    if (!e || engram_expert_grad_open(&g, e, B) != ENGRAM_OK) { ET_CHECK(0); goto out; }
    set_pattern(e);
    for (b = 0; b < B; b++) pos[b] = (b * 11u + 1u) % GRAD_TEXT_N;
    t0 = engram_now_ns();
    ET_OK(engram_expert_backward(e, g, grad_text, GRAD_TEXT_N, pos, B, &bits));
    dt = engram_now_ns() - t0;
    printf("       default expert, a batch of %zu: %.1f us a position (forward + backward, one thread)\n", B,
           (double)dt / 1e3 / (double)B);
    engram_expert_grad_get(g, &gv);
    h = engram_mix2(engram_expert_fingerprint(e), dbits(bits));
    h = engram_mix2(h, fnv_floats(FNV0, gv.bias, 256u));
    h = engram_mix2(h, fnv_floats(FNV0, gv.gain, gv.n_gain));
    h = engram_mix2(h, fnv_floats(FNV0, gv.head, 256u * 512u));
    for (j = 0; j < ENGRAM_EXPERT_MAX_HIDDEN && gv.hidden[j]; j++) h = engram_mix2(h, fnv_floats(FNV0, gv.hidden[j], 512u * 512u));
    h = engram_mix2(h, gv.n_feat);
    h = engram_mix2(h, fnv_floats(FNV0, gv.bag, gv.n_feat * 512u));
    printf("GRADPRINT %016llX\n", (unsigned long long)h);
out:
    engram_expert_grad_close(g);
    engram_expert_close(e);
}

int main(void)
{
    printf("ENGRAM P2.2 -- the slow store learns: the backward pass\n");
    ET_SELFTEST();
    test_reference();
    test_loss_is_bits();
    test_finite_differences();
    test_algebra();
    test_contract();
    test_speed_and_print();
    return et_report("test_learn");
}
