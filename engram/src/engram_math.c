/* ==================================================================================================
 * engram_math.c -- exp and log2 from + - * / and exact conversions. Contract in engram_math.h.
 * ==================================================================================================
 * exp:  x = k ln2 + r, |r| <= ln2 / 2, with ln2 split in two (Cody & Waite 1980) so k ln2 is exact to
 *       well past double precision; e^r by its Taylor series to r^13 / 13! (the first omitted term is
 *       below 2^-60 of the result); then an exact multiplication by 2^k.
 * log2: x = m 2^e, m in [sqrt(1/2), sqrt(2)); ln m = 2 atanh(s), s = (m - 1) / (m + 1), |s| <= 0.172,
 *       by its series to s^25 (the first omitted term is below 2^-64).
 * The coefficients are exact quotients of integers, rounded once by the compiler -- the same double on
 * every conforming compiler, since C99 requires decimal floating constants to round correctly under
 * the default mode (F.7.2).
 * ============================================================================================== */
#include "engram_math.h"

#include <string.h>

static double engram_bits_to_double(uint64_t b)
{
    double d;
    memcpy(&d, &b, sizeof d);
    return d;
}

static uint64_t engram_double_to_bits(double d)
{
    uint64_t b;
    memcpy(&b, &d, sizeof b);
    return b;
}

double engram_pow2i(int k)
{
    if (k > 1023) return engram_bits_to_double(0x7FF0000000000000ull);
    if (k >= -1022) return engram_bits_to_double((uint64_t)(k + 1023) << 52);
    if (k >= -1074) return engram_bits_to_double((uint64_t)1 << (k + 1074));   /* subnormal, exact */
    return 0.0;
}

double engram_round(double x)
{
    /* t = i + f with i = trunc(t) and f = t - i, both EXACT for |t| < 2^52 (the fraction of a double
     * is representable); then compare f with one half. Adding 0.5 first would be wrong: for the largest
     * double below one half, t + 0.5 rounds up to 1. */
    double t = x < 0.0 ? -x : x, r;
    int64_t i = (int64_t)t;
    r = (double)i;
    if (t - r >= 0.5) r += 1.0;
    return x < 0.0 ? -r : r;
}

/* e^r for |r| <= ln2 / 2 + a little: Taylor to r^13 / 13!, Horner */
static double engram_exp_poly(double r)
{
    double p = 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    return p * r + 1.0;
}

/* p * 2^k, in two exact steps when 2^k alone would leave the normal range */
static double engram_scale(double p, int k)
{
    if (k < -1021) return (p * engram_pow2i(k + 1000)) * engram_pow2i(-1000);
    if (k > 1023) return (p * engram_pow2i(k - 1)) * 2.0;
    return p * engram_pow2i(k);
}

double engram_exp(double x)
{
    static const double LN2_HI = 6.93147180369123816490e-01;   /* 0x3FE62E42FEE00000: 32 bits */
    static const double LN2_LO = 1.90821492927058770002e-10;
    static const double INV_LN2 = 1.44269504088896338700e+00;
    double kd;
    if (x != x) return x;
    if (x > 709.782712893384) return engram_bits_to_double(0x7FF0000000000000ull);
    if (x < -745.1332191019412) return 0.0;
    kd = engram_round(x * INV_LN2);
    /* kd has at most 11 significant bits and LN2_HI 32, so kd * LN2_HI is exact */
    return engram_scale(engram_exp_poly((x - kd * LN2_HI) - kd * LN2_LO), (int)kd);
}

double engram_exp2(double x)
{
    static const double LN2 = 6.93147180559945286227e-01;
    double kd;
    if (x != x) return x;
    if (x >= 1024.0) return engram_bits_to_double(0x7FF0000000000000ull);
    if (x < -1075.0) return 0.0;
    kd = engram_round(x);                             /* x - kd is exact: |x| < 2^11 */
    return engram_scale(engram_exp_poly((x - kd) * LN2), (int)kd);
}

double engram_log2(double x)
{
    static const double INV_LN2 = 1.44269504088896338700e+00;
    uint64_t b;
    int e;
    double m, s, s2, t;
    if (x != x || x < 0.0) return engram_bits_to_double(0x7FF8000000000000ull);
    if (x == 0.0) return engram_bits_to_double(0xFFF0000000000000ull);
    b = engram_double_to_bits(x);
    if ((b >> 52) == 0x7FFu) return x;                /* +inf */
    e = (int)(b >> 52);
    if (e == 0) {                                     /* subnormal: scale into the normal range, exactly */
        b = engram_double_to_bits(x * engram_pow2i(64));
        e = (int)(b >> 52) - 64;
    }
    e -= 1023;
    m = engram_bits_to_double((b & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull);    /* [1, 2) */
    if (m > 1.4142135623730951) { m *= 0.5; e++; }
    s = (m - 1.0) / (m + 1.0);
    s2 = s * s;
    t = 1.0 / 25.0;
    t = t * s2 + 1.0 / 23.0;
    t = t * s2 + 1.0 / 21.0;
    t = t * s2 + 1.0 / 19.0;
    t = t * s2 + 1.0 / 17.0;
    t = t * s2 + 1.0 / 15.0;
    t = t * s2 + 1.0 / 13.0;
    t = t * s2 + 1.0 / 11.0;
    t = t * s2 + 1.0 / 9.0;
    t = t * s2 + 1.0 / 7.0;
    t = t * s2 + 1.0 / 5.0;
    t = t * s2 + 1.0 / 3.0;
    t = t * s2 + 1.0;
    return (double)e + (2.0 * s * t) * INV_LN2;
}
