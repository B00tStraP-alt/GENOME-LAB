/* ==================================================================================================
 * engram_expert.c -- one expert of the slow store. Contract, and the forward pass exactly, in
 * engram_expert.h.
 * ============================================================================================== */
#include "engram_expert.h"
#include "engram_alloc.h"
#include "engram_math.h"

#include <math.h>
#include <string.h>

struct engram_expert {
    engram_expert_cfg cfg;
    uint32_t          F;                              /* feature space: orders * buckets       */
    engram_cascade    bag;                            /* F x W, feature-major                  */
    engram_cascade    hid[ENGRAM_EXPERT_MAX_HIDDEN];  /* W x W                                  */
    engram_cascade    head;                           /* 256 x W                                */
    float            *gain;                           /* (hidden + 1) * W + 256                 */
    float            *bias;                           /* 256                                    */
    size_t            n_gain;
    /* forward scratch for one position */
    int32_t          *zi;                             /* max(W, 256)                            */
    float            *act;                            /* W                                      */
    int8_t           *q;                              /* W                                      */
    float            *logit;                          /* 256                                    */
};

void engram_expert_cfg_default(engram_expert_cfg *c)
{
    if (!c) return;
    memset(c, 0, sizeof *c);
    engram_ctx_cfg_default(&c->ctx);
    c->width = 512u;
    c->hidden = 0u;
    c->K = 16u;
    c->kappa = 0.5f;
    c->lr = 0.1f;
    c->lr_unit = 2e-3f;
    c->seed = 0x534C4F57ull;                          /* "SLOW" */
}

/* weights the configuration asks for, or 0 on overflow */
static size_t engram_expert_weights(const engram_expert_cfg *c)
{
    size_t F = (size_t)engram_ctx_features(&c->ctx), W = c->width, total;
    total = F * W;                                    /* F <= 2^24, W <= 4096: no overflow in 64 bits */
    total += (size_t)c->hidden * W * W;
    total += (size_t)ENGRAM_EXPERT_VOCAB * W;
    return total;
}

engram_rc engram_expert_cfg_check(const engram_expert_cfg *c)
{
    if (!c || engram_ctx_cfg_check(&c->ctx) != ENGRAM_OK) return ENGRAM_E_ARG;
    if (c->width < 16u || c->width > 4096u || c->width % 16u || c->hidden > ENGRAM_EXPERT_MAX_HIDDEN)
        return ENGRAM_E_ARG;
    if (c->K != 4u && c->K != 16u && c->K != 64u) return ENGRAM_E_ARG;
    if (!(c->kappa >= 0.0f && c->kappa <= 16.0f) || !(c->lr > 0.0f && c->lr <= 64.0f) ||
        !(c->lr_unit > 0.0f && c->lr_unit <= 1.0f))
        return ENGRAM_E_ARG;
    if (engram_expert_weights(c) > ENGRAM_EXPERT_MAX_WEIGHTS) return ENGRAM_E_FULL;
    return ENGRAM_OK;
}

void engram_expert_close(engram_expert *e)
{
    unsigned i;
    if (!e) return;
    engram_cascade_free(&e->bag);
    for (i = 0; i < ENGRAM_EXPERT_MAX_HIDDEN; i++) engram_cascade_free(&e->hid[i]);
    engram_cascade_free(&e->head);
    engram_free(e->gain); engram_free(e->bias);
    engram_free(e->zi); engram_free(e->act); engram_free(e->q); engram_free(e->logit);
    engram_free(e);
}

engram_rc engram_expert_open(engram_expert **out, const engram_expert_cfg *cfg)
{
    engram_expert_cfg d;
    engram_expert *e;
    unsigned i, W;
    size_t k;
    engram_rc rc;
    if (!out) return ENGRAM_E_ARG;
    *out = NULL;
    if (!cfg) { engram_expert_cfg_default(&d); cfg = &d; }
    if ((rc = engram_expert_cfg_check(cfg)) != ENGRAM_OK) return rc;
    e = (engram_expert *)engram_calloc(1u, sizeof *e);
    if (!e) return ENGRAM_E_MEM;
    e->cfg = *cfg;
    W = cfg->width;
    e->F = engram_ctx_features(&cfg->ctx);
    /* each matrix its own seed, derived: the same expert from the same cfg, on every platform */
    rc = engram_cascade_init(&e->bag, e->F, W, cfg->K, cfg->kappa, engram_mix2(cfg->seed, 1u));
    for (i = 0; rc == ENGRAM_OK && i < cfg->hidden; i++)
        rc = engram_cascade_init(&e->hid[i], W, W, cfg->K, cfg->kappa, engram_mix2(cfg->seed, 2u + i));
    if (rc == ENGRAM_OK)
        rc = engram_cascade_init(&e->head, ENGRAM_EXPERT_VOCAB, W, cfg->K, cfg->kappa, engram_mix2(cfg->seed, 99u));
    if (rc != ENGRAM_OK) { engram_expert_close(e); return rc; }
    e->n_gain = (size_t)(cfg->hidden + 1u) * W + ENGRAM_EXPERT_VOCAB;
    e->gain = (float *)engram_array(e->n_gain, sizeof *e->gain);
    e->bias = (float *)engram_calloc(ENGRAM_EXPERT_VOCAB, sizeof *e->bias);
    e->zi = (int32_t *)engram_array(W > ENGRAM_EXPERT_VOCAB ? W : ENGRAM_EXPERT_VOCAB, sizeof *e->zi);
    e->act = (float *)engram_array(W, sizeof *e->act);
    e->q = (int8_t *)engram_array(W, sizeof *e->q);
    e->logit = (float *)engram_array(ENGRAM_EXPERT_VOCAB, sizeof *e->logit);
    if (!e->gain || !e->bias || !e->zi || !e->act || !e->q || !e->logit) { engram_expert_close(e); return ENGRAM_E_MEM; }
    for (k = 0; k < e->n_gain; k++) e->gain[k] = 1.0f;
    *out = e;
    return ENGRAM_OK;
}

const engram_expert_cfg *engram_expert_cfg_get(const engram_expert *e)
{
    return e ? &e->cfg : NULL;
}

/* norm(z, scale) of engram_expert.h */
static void engram_norm_i(const int32_t *z, unsigned n, double scale, float *out)
{
    double ss = 0.0, r;
    unsigned i;
    for (i = 0; i < n; i++) { double v = (double)z[i] * scale; ss += v * v; }
    r = sqrt(ss / (double)n + 1e-6);
    for (i = 0; i < n; i++) out[i] = (float)((double)z[i] * scale / r);
}

/* act(a, gain) of engram_expert.h; returns s */
static double engram_act_quant(float *a, const float *gain, unsigned n, int8_t *q)
{
    float mx = 0.0f;
    double s;
    unsigned i;
    for (i = 0; i < n; i++) {
        float y = a[i] * gain[i];
        a[i] = y > 0.0f ? y : 0.0f;
        if (a[i] > mx) mx = a[i];
    }
    s = (double)mx / 127.0 + 1e-12;
    for (i = 0; i < n; i++) q[i] = (int8_t)engram_round((double)a[i] / s);
    return s;
}

/* the forward pass for position t, into e->logit */
static void engram_expert_forward(engram_expert *e, const uint8_t *text, size_t n, size_t t)
{
    const unsigned W = e->cfg.width;
    const double alpha = 1.0 / sqrt((double)W);
    uint32_t ids[ENGRAM_CTX_MAX_ORDERS];
    double s;
    unsigned j, v;
    engram_ctx_ids(&e->cfg.ctx, text, n, t, ids);
    /* the ids come from engram_ctx for this expert's own cfg: always inside the bag */
    (void)engram_cascade_gather(&e->bag, ids, e->cfg.ctx.orders, e->zi);
    engram_norm_i(e->zi, W, 1.0 / sqrt((double)e->cfg.ctx.orders), e->act);
    s = engram_act_quant(e->act, e->gain, W, e->q);
    for (j = 0; j < e->cfg.hidden; j++) {
        engram_cascade_matvec(&e->hid[j], e->q, e->zi, 0u, W);
        engram_norm_i(e->zi, W, s * alpha, e->act);
        s = engram_act_quant(e->act, e->gain + (size_t)(j + 1u) * W, W, e->q);
    }
    engram_cascade_matvec(&e->head, e->q, e->zi, 0u, ENGRAM_EXPERT_VOCAB);
    engram_norm_i(e->zi, ENGRAM_EXPERT_VOCAB, s * alpha, e->logit);
    for (v = 0; v < ENGRAM_EXPERT_VOCAB; v++)
        e->logit[v] = e->logit[v] * e->gain[(size_t)(e->cfg.hidden + 1u) * W + v] + e->bias[v];
}

engram_rc engram_expert_logits(engram_expert *e, const uint8_t *text, size_t n, size_t t, float logits[256])
{
    if (!e || !logits || (!text && n) || t > n) return ENGRAM_E_ARG;
    engram_expert_forward(e, text, n, t);
    memcpy(logits, e->logit, ENGRAM_EXPERT_VOCAB * sizeof *logits);
    return ENGRAM_OK;
}

engram_rc engram_expert_bits(engram_expert *e, const uint8_t *text, size_t n, size_t from, size_t to,
                             double *bits)
{
    static const double LOG2E = 1.44269504088896338700;
    size_t t;
    double total = 0.0;
    if (bits) *bits = 0.0;
    if (!e || !bits || (!text && n) || from > to || to > n) return ENGRAM_E_ARG;
    for (t = from; t < to; t++) {
        double m, z = 0.0;
        unsigned v;
        engram_expert_forward(e, text, n, t);
        m = (double)e->logit[0];
        for (v = 1; v < ENGRAM_EXPERT_VOCAB; v++) if ((double)e->logit[v] > m) m = (double)e->logit[v];
        for (v = 0; v < ENGRAM_EXPERT_VOCAB; v++) z += engram_exp((double)e->logit[v] - m);
        total += engram_log2(z) - ((double)e->logit[text[t]] - m) * LOG2E;
    }
    *bits = total;
    return ENGRAM_OK;
}

void engram_expert_units_get(engram_expert *e, engram_expert_units *u)
{
    if (!u) return;
    memset(u, 0, sizeof *u);
    if (!e) return;
    u->gain = e->gain;
    u->n_gain = e->n_gain;
    u->bias = e->bias;
    u->n_bias = ENGRAM_EXPERT_VOCAB;
}

void engram_expert_stats_get(const engram_expert *e, engram_expert_stats *st)
{
    size_t W;
    if (!st) return;
    memset(st, 0, sizeof *st);
    if (!e) return;
    W = e->cfg.width;
    st->weights = engram_expert_weights(&e->cfg);
    st->units = e->n_gain + ENGRAM_EXPERT_VOCAB;
    /* positions and the value plane: two bytes a weight */
    st->bytes_resident = sizeof *e + 2u * st->weights + st->units * sizeof(float) +
                         (W > ENGRAM_EXPERT_VOCAB ? W : ENGRAM_EXPERT_VOCAB) * sizeof(int32_t) + W * sizeof(float) + W +
                         ENGRAM_EXPERT_VOCAB * sizeof(float);
}

uint64_t engram_expert_fingerprint(const engram_expert *e)
{
    uint64_t h;
    uint32_t kb, lb, ub;
    unsigned i;
    if (!e) return 0u;
    memcpy(&kb, &e->cfg.kappa, sizeof kb);
    memcpy(&lb, &e->cfg.lr, sizeof lb);
    memcpy(&ub, &e->cfg.lr_unit, sizeof ub);
    h = engram_mix2(0x455850455254ull, ((uint64_t)e->cfg.ctx.orders << 32) | e->cfg.ctx.buckets_log2);
    h = engram_mix2(h, e->cfg.ctx.seed);
    h = engram_mix2(h, ((uint64_t)e->cfg.width << 32) | ((uint64_t)e->cfg.hidden << 16) | e->cfg.K);
    h = engram_mix2(h, ((uint64_t)kb << 32) | lb);
    h = engram_mix2(h, ((uint64_t)ub << 32));
    h = engram_mix2(h, e->cfg.seed);
    h = engram_mix2(h, engram_cascade_fingerprint(&e->bag));
    for (i = 0; i < e->cfg.hidden; i++) h = engram_mix2(h, engram_cascade_fingerprint(&e->hid[i]));
    h = engram_mix2(h, engram_cascade_fingerprint(&e->head));
    h = engram_mix2(h, engram_hash_bytes(e->gain, e->n_gain * sizeof *e->gain, 1u));
    return engram_mix2(h, engram_hash_bytes(e->bias, ENGRAM_EXPERT_VOCAB * sizeof *e->bias, 2u));
}
