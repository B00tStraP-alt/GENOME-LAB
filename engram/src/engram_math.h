/* ==================================================================================================
 * engram_math.h -- the transcendental functions ENGRAM computes, from + - * / alone.
 * ==================================================================================================
 *
 * WHY NOT <math.h>
 * ==================================================================================================
 * C99 requires sqrt to be correctly rounded and says nothing of exp or log: glibc and the Windows CRT
 * are free to differ in the last bit, and they do (W-P1.1-4). The slow store (Phase 2) trains weights
 * whose every update depends on a softmax, and its training must give the same bits on every machine
 * (R5) -- a single last-bit difference in one exp, fed back through a million updates, is a different
 * model. So the softmax and the loss are computed here, from IEEE-754 + - * / and exact conversions
 * only, under -ffp-contract=off: the same operations in the same order, so the same bits, everywhere.
 *
 * ACCURACY (test_math, against the host libm over 20 million arguments): engram_exp and engram_exp2
 * within 1 ulp, engram_log2 within 4. They are not faster than libm and do not try to be: they are the
 * same everywhere, which libm is not.
 * ============================================================================================== */
#ifndef ENGRAM_MATH_H
#define ENGRAM_MATH_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* e^x. +inf above 709.78, 0 below -745.2, NaN for NaN. Subnormal results are produced, not flushed. */
double engram_exp(double x);

/* 2^x, as engram_exp(x * ln 2) with the product split so no bit of x is lost. */
double engram_exp2(double x);

/* log2(x) for finite x > 0 (subnormals included). -inf for 0, NaN for x < 0 or NaN, +inf for +inf. */
double engram_log2(double x);

/* 2^k exactly, k in [-1074, 1023]; 0 below, +inf above. */
double engram_pow2i(int k);

/* The value nearest x, halfway cases away from zero, computed without the rounding mode: the same on
 * every platform whatever the FPU is set to. |x| < 2^52. */
double engram_round(double x);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_MATH_H */
