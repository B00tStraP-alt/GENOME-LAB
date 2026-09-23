/* ==================================================================================================
 * engram_rng.h -- reproducible randomness. The same seed gives the same bytes on every machine.
 * ==================================================================================================
 *
 * xoshiro256** (Blackman & Vigna 2018), seeded through splitmix64. Fast, 256 bits of state, passes
 * BigCrush, and -- the property that matters here -- its output is a pure function of integer
 * arithmetic, so it is bit-identical on every compiler and CPU.
 *
 * THIS IS NOT FOR KEYS. Key material comes from engram_os_random(). This generator exists so that
 * experiments are repeatable: a weight initialisation, a shuffle, an arm's replay order must come out
 * the same on the second run, or the difference between two arms is partly the seed and no longer a
 * measurement (rule R4).
 *
 * ==================================================================================================
 * WHY THERE IS NO BOX-MULLER NORMAL
 * ==================================================================================================
 * A Box-Muller or polar normal needs log(). The C standard does not require log() to be correctly
 * rounded, and glibc and the Windows CRT do not agree to the last bit on every input. A weight
 * initialised through log() on Linux would therefore differ from the same weight initialised on
 * Windows -- not by much, and not detectably by any test that compares within a tolerance, but enough
 * that a store written on one reads back through a different initialisation on the other.
 *
 * sqrt() IS required by IEEE-754 to be correctly rounded, so it is safe. Everything here is built on
 * integers, sqrt, and the four basic operations, which is exactly the set every conforming platform
 * computes identically -- PROVIDED the build forbids FMA contraction (-ffp-contract=off), since a fused
 * multiply-add rounds once where a*b+c rounds twice.
 *
 * The normal offered instead is Irwin-Hall: the sum of twelve uniforms, minus six. Mean 0, variance 1,
 * bounded to [-6, 6], no transcendentals. Its tails are thinner than a true Gaussian's, which for
 * initialisation and noise injection is a feature -- an outlier weight six sigma from the mean is not
 * something an initialiser should ever produce.
 * ============================================================================================== */
#ifndef ENGRAM_RNG_H
#define ENGRAM_RNG_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { uint64_t s[4]; } engram_rng;

/* Any seed is valid, including 0: splitmix64 expands it so the state is never all zeros. */
void     engram_rng_seed(engram_rng *r, uint64_t seed);

uint64_t engram_rng_u64(engram_rng *r);
uint32_t engram_rng_u32(engram_rng *r);

/* Uniform on [0, n), UNBIASED (rejection, not modulo -- modulo favours small residues whenever n does
 * not divide 2^64). n == 0 returns 0. */
uint64_t engram_rng_below(engram_rng *r, uint64_t n);

double   engram_rng_unit(engram_rng *r);     /* [0, 1), 53 bits: every representable step reachable */
float    engram_rng_unitf(engram_rng *r);    /* [0, 1), 24 bits                                     */
double   engram_rng_range(engram_rng *r, double lo, double hi);   /* [lo, hi)                       */

/* Irwin-Hall approximate standard normal: mean 0, variance 1, support [-6, 6]. Bit-reproducible. */
double   engram_rng_normal(engram_rng *r);

/* Fisher-Yates, uniform over all n! orderings. */
void     engram_rng_shuffle_u32(engram_rng *r, uint32_t *a, size_t n);

/* Advance by 2^128 steps. Calling this k times from one seed yields k NON-OVERLAPPING streams of 2^128
 * values each -- how parallel workers get independent randomness that is still a pure function of
 * the master seed, so a threaded run reproduces exactly (P2.2). */
void     engram_rng_jump(engram_rng *r);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_RNG_H */
