/* ==================================================================================================
 * test_math.c -- P2.1 (part 1): exp, exp2, log2 and round from + - * / alone.
 * ==================================================================================================
 *   M1  special values: NaN, infinities, zero, the overflow and underflow edges, subnormals
 *   M2  accuracy against the host libm over millions of arguments -- a reference that is NOT the thing
 *       being tested, whose last bit may differ from platform to platform (which is the point)
 *   M3  identities that do not need libm: exp(0) = 1, exp2(k) = 2^k exactly, log2(2^k) = k exactly,
 *       exp2(log2(x)) ~ x, monotonicity on a fine grid
 *   M4  round: half away from zero, the largest double below 0.5, every integer boundary
 *   M5  MATHPRINT: a hash of the bits over a fixed grid -- the gate requires Linux and Windows to agree,
 *       which libm itself does not promise
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static uint64_t bits(double d)
{
    uint64_t b;
    memcpy(&b, &d, sizeof b);
    return b;
}

static double ulps(double a, double b)
{
    int64_t ia, ib;
    if (a == b) return 0.0;
    if (a != a || b != b) return 1e300;
    memcpy(&ia, &a, 8);
    memcpy(&ib, &b, 8);
    if ((ia < 0) != (ib < 0)) return 1e300;
    return (double)(ia > ib ? ia - ib : ib - ia);
}

static int g_quick;

static void test_special(void)
{
    ET_SECTION("M1: special values -- NaN in, NaN out; the edges of the range; subnormals produced, not flushed");
    ET_CHECK(engram_exp((double)NAN) != engram_exp((double)NAN));
    ET_CHECK(engram_log2((double)NAN) != engram_log2((double)NAN));
    ET_CHECK(engram_log2(-1.0) != engram_log2(-1.0));
    ET_CHECK(engram_exp(HUGE_VAL) == HUGE_VAL);
    ET_CHECK(engram_exp(-HUGE_VAL) == 0.0);
    ET_CHECK(engram_exp(710.0) == HUGE_VAL);
    ET_CHECK(engram_exp(709.78) < HUGE_VAL && engram_exp(709.78) > 1e308);
    ET_CHECK(engram_exp(-746.0) == 0.0);
    ET_CHECK(engram_exp(-745.0) > 0.0);                          /* the smallest subnormal region */
    ET_CHECK(engram_exp(0.0) == 1.0);
    ET_CHECK(engram_exp2(-1074.0) == 4.9406564584124654e-324);
    ET_CHECK(engram_exp2(1024.0) == HUGE_VAL);
    ET_CHECK(engram_exp2(1023.0) == 8.98846567431158e307);
    ET_CHECK(engram_log2(0.0) == -HUGE_VAL);
    ET_CHECK(engram_log2(HUGE_VAL) == HUGE_VAL);
    ET_CHECK(engram_log2(4.9406564584124654e-324) == -1074.0);
    ET_CHECK(engram_log2(1.0) == 0.0);
    ET_CHECK(engram_pow2i(-1075) == 0.0 && engram_pow2i(1024) == HUGE_VAL && engram_pow2i(0) == 1.0);
    ET_CHECK(engram_pow2i(-1074) == 4.9406564584124654e-324);
}

static void test_accuracy(void)
{
    double we = 0, w2 = 0, wl = 0, wex = 0, w2x = 0, wlx = 0;
    uint64_t s = 0x9E3779B97F4A7C15ull;
    long i, n = g_quick ? 2000000L : 20000000L;
    ET_SECTION("M2: against the host libm -- exp and exp2 within 1 ulp, log2 within 4, over the whole range");
    for (i = 0; i < n; i++) {
        double u, x, e;
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        u = (double)(s >> 11) / 9007199254740992.0;
        x = (i % 4 == 0) ? -2.0 + 4.0 * u : -745.0 + u * 1454.7;
        e = ulps(engram_exp(x), exp(x));
        if (e > we) { we = e; wex = x; }
        x = (i % 4 == 0) ? -2.0 + 4.0 * u : -1074.0 + u * 2097.9;
        e = ulps(engram_exp2(x), exp2(x));
        if (e > w2) { w2 = e; w2x = x; }
        x = (i % 3 == 0) ? 0.5 + u : ldexp(0.5 + u, (int)(u * 2098.0) - 1074);
        if (fabs(log2(x)) > 1e-2) {
            e = ulps(engram_log2(x), log2(x));
            if (e > wl) { wl = e; wlx = x; }
        }
    }
    ET_CHECKF(we <= 1.0, "exp: %.0f ulp at %.17g", we, wex);
    ET_CHECKF(w2 <= 1.0, "exp2: %.0f ulp at %.17g", w2, w2x);
    ET_CHECKF(wl <= 4.0, "log2: %.0f ulp at %.17g", wl, wlx);
    printf("       %ld arguments: exp %.0f ulp, exp2 %.0f ulp, log2 %.0f ulp at worst\n", n, we, w2, wl);
}

static void test_identities(void)
{
    int k;
    double x, prev;
    unsigned bad = 0, mono = 0;
    ET_SECTION("M3: exact identities -- exp2(k) = 2^k and log2(2^k) = k for every k; monotone on a fine grid");
    for (k = -1074; k <= 1023; k++) {
        if (engram_exp2((double)k) != engram_pow2i(k)) bad++;
        if (engram_log2(engram_pow2i(k)) != (double)k) bad++;
        if (k >= -1022 && engram_pow2i(k) != ldexp(1.0, k)) bad++;
    }
    ET_EQ_U64(bad, 0u);
    prev = 0.0;
    for (x = -20.0; x <= 20.0; x += 1.0 / 4096.0) {
        double e = engram_exp(x);
        if (e < prev) mono++;
        prev = e;
    }
    prev = -HUGE_VAL;
    for (x = 1e-3; x <= 1e3; x *= 1.0009765625) {
        double l = engram_log2(x);
        if (l < prev) mono++;
        prev = l;
    }
    ET_EQ_U64(mono, 0u);
    bad = 0;
    for (x = 1e-6; x <= 1e6; x *= 1.37) if (fabs(engram_exp2(engram_log2(x)) / x - 1.0) > 1e-14) bad++;
    ET_EQ_U64(bad, 0u);
}

static void test_round(void)
{
    ET_SECTION("M4: round -- halves away from zero, the largest double below one half, integer boundaries");
    ET_CHECK(engram_round(0.5) == 1.0 && engram_round(-0.5) == -1.0);
    ET_CHECK(engram_round(1.5) == 2.0 && engram_round(2.5) == 3.0 && engram_round(-2.5) == -3.0);
    ET_CHECK(engram_round(0.49999999999999994) == 0.0);
    ET_CHECK(engram_round(-0.49999999999999994) == 0.0);
    ET_CHECK(engram_round(4503599627370495.5) == 4503599627370496.0);
    ET_CHECK(engram_round(126.99999) == 127.0 && engram_round(-127.4) == -127.0);
    {
        int i;
        unsigned bad = 0;
        for (i = -100000; i <= 100000; i++) {
            double x = (double)i / 64.0;
            double want = x < 0 ? -floor(-x + 0.5) : floor(x + 0.5);
            if (engram_round(x) != want) bad++;
        }
        ET_EQ_U64(bad, 0u);
    }
}

static void test_print(void)
{
    uint64_t h = 0x4D415448ull;
    int i;
    ET_SECTION("M5: MATHPRINT -- the bits over a fixed grid, compared across platforms by the gate");
    for (i = 0; i < 200000; i++) {
        double x = -700.0 + (double)i * (1400.0 / 200000.0);
        h = engram_mix2(h, bits(engram_exp(x)));
        h = engram_mix2(h, bits(engram_exp2(x * 1.4)));
        h = engram_mix2(h, bits(engram_log2(engram_pow2i(-1000 + i % 2000) * (1.0 + (double)i / 200000.0))));
    }
    printf("MATHPRINT %016llX\n", (unsigned long long)h);
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = q && q[0] == '1';
    printf("ENGRAM P2.1 -- math\n");
    ET_SELFTEST();
    test_special();
    test_accuracy();
    test_identities();
    test_round();
    test_print();
    return et_report("test_math");
}
