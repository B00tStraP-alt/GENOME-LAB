/* ==================================================================================================
 * engram_cascade.h -- THE CASCADE WEIGHT: a ternary weight, its whole training state, and how
 * consolidated it is, in one byte.
 * ==================================================================================================
 *
 * THE WORD (research/cascade_weight/README.md has the measurements)
 * ==================================================================================================
 * Three basins of K positions on one line; a weight is a position p in [0, 3K):
 *
 *      p        0 ........ K-1 | K ........ 2K-1 | 2K ........ 3K-1
 *      value         -1        |        0         |        +1
 *      depth    K-1 ......   0 | 0 (lean) ... K-1 | 0 ......   K-1
 *
 * VALUE is what inference reads (it ships at 1.6 bits: five trits a byte). DEPTH is how far the
 * weight is from changing its value -- and, because moves toward zero are slowed by depth, how
 * CONSOLIDATED it is. K = 16 (6 bits of state) is the measured default; the byte holds K up to 85.
 *
 * THE UPDATE -- no float is kept per weight
 * ==================================================================================================
 * For a row of the matrix, the transient gradient g becomes a desired move in positions
 *      u = -lr * g / rms(g over the row)
 * capped at K (one basin a step). A move that brings a +-1 closer to 0 is first scaled by
 *      2^(-kappa * depth * 4 / K)                         metaplasticity (Fusi, Drew & Abbott 2005)
 * and the move taken is floor(|u|) plus one more with probability frac(|u|): its EXPECTED value is
 * exactly u, so small gradients accumulate instead of vanishing. The randomness is a counter hash of
 * (seed, step, index) -- splitmix64's finaliser, engram_mix64 -- so a draw depends on WHICH weight and
 * WHICH step, never on the order weights are visited or the thread that visits them: an update split
 * over any number of threads gives the same bytes (R5), and the same bytes as the numpy prototype.
 * ============================================================================================== */
#ifndef ENGRAM_CASCADE_H
#define ENGRAM_CASCADE_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_CASCADE_KMAX 64u

typedef struct {
    uint32_t rows, cols;
    uint32_t K;                          /* positions per basin: 4, 16 or 64                          */
    float    kappa;                      /* metaplasticity; 0 disables it (the control)               */
    uint64_t seed;
    uint64_t step;                       /* updates applied: part of every draw's key                 */
    uint8_t *pos;                        /* rows * cols positions, row-major                          */
    int8_t  *val;                        /* the value plane, kept equal to value(pos) after every update */
    float    meta[ENGRAM_CASCADE_KMAX];  /* 2^(-kappa * d * 4 / K) for d = 0 .. K-1                   */
} engram_cascade;

/* A fresh matrix: every position drawn uniformly from [0, 3K) by the counter hash with step
 * 0xFFFFFFFF (the prototype's initialisation, so the two agree bit for bit). */
engram_rc engram_cascade_init(engram_cascade *m, uint32_t rows, uint32_t cols, uint32_t K, float kappa,
                              uint64_t seed);
void      engram_cascade_free(engram_cascade *m);

static inline int engram_cascade_value(uint32_t p, uint32_t K)
{
    return (p >= 2u * K) - (p < K);
}

static inline uint32_t engram_cascade_depth(uint32_t p, uint32_t K)
{
    return p < K ? K - 1u - p : (p >= 2u * K ? p - 2u * K : p - K);
}

/* The uniform in [0, 1), 24 bits, exact in a float, for weight `index` at update `step`. */
float engram_cascade_uniform(uint64_t seed, uint64_t step, uint64_t index);

/* Apply one update to rows [r0, r1) from the gradient g (rows x cols, row-major; only those rows are
 * read). `step` is the update's number -- the caller passes m->step + 1 to every row range of one
 * update, then calls engram_cascade_commit_step. *moved counts the weights whose position changed,
 * *flipped those whose VALUE changed (the bits of the inference plane that moved -- P3.3 accounts
 * for them). Allocates nothing, cannot fail. */
void engram_cascade_update_rows(engram_cascade *m, const float *g, float lr, uint64_t step, uint32_t r0,
                                uint32_t r1, uint64_t *moved, uint64_t *flipped);
void engram_cascade_commit_step(engram_cascade *m, uint64_t step);

/* The same update for a matrix STORED transposed: m holds rows = FEATURES (F) x cols = UNITS (H) --
 * the layout a bag of embedding rows is read in -- while the update treats it as the H x F matrix it
 * is: each unit's gradient is normalised over ALL F features (zeros counted in the mean) and the draw
 * for (unit h, feature f) is keyed h * F + f. Only the listed features (ascending, distinct) carry a
 * gradient: g is n_feat x H, row i the gradient of feature feat[i]. Units [h0, h1) are updated -- the
 * partition threads split. (research/p2_slow: normalising per unit, not per feature, 2.43 vs 2.60 bits
 * per byte -- it keeps how much evidence each feature had in the batch.) */
void engram_cascade_update_units(engram_cascade *m, const uint32_t *feat, size_t n_feat, const float *g, float lr,
                                 uint64_t step, uint32_t h0, uint32_t h1, uint64_t *moved, uint64_t *flipped);

/* ---- the products the network needs, over the VALUE plane; integer, so EXACT on every platform ---- */
/* y[r] = sum_c value[r][c] * x[c], r in [r0, r1). |y| <= cols * 127, well inside int32 for cols < 2^24. */
void engram_cascade_matvec(const engram_cascade *m, const int8_t *x, int32_t *y, uint32_t r0, uint32_t r1);

/* y[c] = sum_r value[r][c] * d[r] for c in [c0, c1): the transposed product backprop needs. Floats,
 * summed over r in index order -- the same order whatever the partition of c. */
void engram_cascade_matvec_t(const engram_cascade *m, const float *d, float *y, uint32_t c0, uint32_t c1);

/* y[c] = sum over the n listed rows of value[row][c]: a bag of embedding rows. Rows may repeat.
 * ENGRAM_E_ARG, y untouched, if any row is outside the matrix -- checked before a single add. */
engram_rc engram_cascade_gather(const engram_cascade *m, const uint32_t *rows, size_t n, int32_t *y);

/* ---- persistence: shape, K, kappa, seed, step, then the positions (one byte each) ----------------- */
/* The serialised size: 4 x u32 + f32 + 2 x u64 + rows * cols bytes. */
size_t    engram_cascade_size(const engram_cascade *m);
/* Write into out (engram_cascade_size bytes). */
void      engram_cascade_write(const engram_cascade *m, uint8_t *out);
/* Read and PROVE a matrix: shape within the given bounds, K one of 4 / 16 / 64, kappa finite in [0, 16],
 * every position < 3K. ENGRAM_E_FORMAT otherwise; *used = bytes consumed. m is initialised (and owns
 * memory) only on ENGRAM_OK. */
engram_rc engram_cascade_read(engram_cascade *m, const uint8_t *p, size_t n, uint32_t max_rows, uint32_t max_cols,
                              size_t *used);

/* A 64-bit hash of shape, K, kappa, seed, step and every position. */
uint64_t engram_cascade_fingerprint(const engram_cascade *m);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_CASCADE_H */
