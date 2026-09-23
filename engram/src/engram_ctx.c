/* ==================================================================================================
 * engram_ctx.c -- hashed context features. Contract in engram_ctx.h.
 * ============================================================================================== */
#include "engram_ctx.h"

#include <string.h>

void engram_ctx_cfg_default(engram_ctx_cfg *c)
{
    if (!c) return;
    memset(c, 0, sizeof *c);
    c->orders = 6u;
    c->buckets_log2 = 12u;
    c->seed = 0x435458ull;                                  /* "CTX" */
}

engram_rc engram_ctx_cfg_check(const engram_ctx_cfg *c)
{
    if (!c || c->orders < 1u || c->orders > ENGRAM_CTX_MAX_ORDERS || c->buckets_log2 < 4u || c->buckets_log2 > 20u)
        return ENGRAM_E_ARG;
    return ENGRAM_OK;
}

uint32_t engram_ctx_features(const engram_ctx_cfg *c)
{
    return (uint32_t)c->orders << c->buckets_log2;
}

void engram_ctx_ids(const engram_ctx_cfg *c, const uint8_t *text, size_t n, size_t t, uint32_t *out)
{
    const uint64_t mask = ((uint64_t)1 << c->buckets_log2) - 1u;
    uint64_t h = c->seed;
    unsigned k;
    if (t > n) t = n;
    for (k = 1; k <= c->orders; k++) {
        uint64_t b = t >= k ? (uint64_t)text[t - k] : 256u;
        h = engram_mix2(h, b);
        out[k - 1u] = (uint32_t)(((uint64_t)(k - 1u) << c->buckets_log2) | (h & mask));
    }
}
