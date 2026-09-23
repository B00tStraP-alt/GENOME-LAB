/* ==================================================================================================
 * engram_ctx.h -- what a slow-store expert sees of the text before the cursor: hashed context features.
 * ==================================================================================================
 *
 * For the byte at position t the expert reads ORDERS features: for k = 1 .. ORDERS, the k bytes before
 * t (t-k .. t-1), hashed. Order k lands in its own block of BUCKETS feature ids, so orders never
 * collide with one another:
 *
 *      h_0 = seed          h_k = mix2(h_{k-1}, byte[t - k])      (256 stands for "before the start")
 *      feature_k = (k - 1) * BUCKETS + (h_k mod BUCKETS)
 *
 * WHY HASHED CONTEXTS AND NOT A WINDOW OF BYTE EMBEDDINGS (research/p2_slow, sweep 1): with every
 * other part of the network the same, an expert reading the last 12 bytes as learned embeddings
 * reached 3.20 bits per byte on held-out English; the same expert reading 6 hashed orders reached 2.43
 * after one epoch -- past an interpolated order-3 count model (2.44). A hashed order-k feature names a
 * CONTEXT, which is what a memory of text is: consolidating an episode into weights is teaching the
 * expert what follows the contexts that episode contains.
 * ============================================================================================== */
#ifndef ENGRAM_CTX_H
#define ENGRAM_CTX_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_CTX_MAX_ORDERS 16u

typedef struct {
    unsigned orders;         /* 1 .. ENGRAM_CTX_MAX_ORDERS                     */
    unsigned buckets_log2;   /* BUCKETS = 2^buckets_log2, 4 .. 20               */
    uint64_t seed;
} engram_ctx_cfg;

void      engram_ctx_cfg_default(engram_ctx_cfg *c);        /* 6 orders, 4096 buckets each */
engram_rc engram_ctx_cfg_check(const engram_ctx_cfg *c);
uint32_t  engram_ctx_features(const engram_ctx_cfg *c);     /* orders * buckets: the feature space */

/* The feature ids of position t in text[0 .. n): out[0 .. orders). t may equal n (the byte after the
 * text). Never fails for a checked cfg; t > n is clamped to n. */
void engram_ctx_ids(const engram_ctx_cfg *c, const uint8_t *text, size_t n, size_t t, uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_CTX_H */
