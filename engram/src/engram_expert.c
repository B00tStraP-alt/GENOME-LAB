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

/* norm(z, scale) of engram_expert.h; returns r */
static double engram_norm_i(const int32_t *z, unsigned n, double scale, float *out)
{
    double ss = 0.0, r;
    unsigned i;
    for (i = 0; i < n; i++) { double v = (double)z[i] * scale; ss += v * v; }
    r = sqrt(ss / (double)n + 1e-6);
    for (i = 0; i < n; i++) out[i] = (float)((double)z[i] * scale / r);
    return r;
}

/* act(x, gain) of engram_expert.h, into a (which may be x) and q; returns s */
static double engram_act_quant(const float *x, const float *gain, unsigned n, float *a, int8_t *q)
{
    float mx = 0.0f;
    double s;
    unsigned i;
    for (i = 0; i < n; i++) {
        float y = x[i] * gain[i];
        a[i] = y > 0.0f ? y : 0.0f;
        if (a[i] > mx) mx = a[i];
    }
    s = (double)mx / 127.0 + 1e-12;
    for (i = 0; i < n; i++) q[i] = (int8_t)engram_round((double)a[i] / s);
    return s;
}

/* Where one forward pass puts what it computes. Layer j = 0 .. hidden is the bag's norm (j = 0) or hidden
 * layer j's; index hidden + 1 is the head. Inference points every layer at the expert's one scratch; the
 * backward pass points each at its own slot, because it needs them all again. */
typedef struct {
    uint32_t ids[ENGRAM_CTX_MAX_ORDERS];
    int32_t *z[ENGRAM_EXPERT_MAX_HIDDEN + 2u];     /* the integer pre-activations                */
    float   *n[ENGRAM_EXPERT_MAX_HIDDEN + 2u];     /* norm(z, sc)                                */
    float   *a[ENGRAM_EXPERT_MAX_HIDDEN + 1u];     /* after gain and ReLU (may be n's buffer)   */
    int8_t  *q[ENGRAM_EXPERT_MAX_HIDDEN + 1u];
    double   sc[ENGRAM_EXPERT_MAX_HIDDEN + 2u];    /* the norm's scale                           */
    double   r[ENGRAM_EXPERT_MAX_HIDDEN + 2u];     /* the norm's root mean square                */
    double   s[ENGRAM_EXPERT_MAX_HIDDEN + 1u];     /* the int8 scale                             */
    float   *logit;                                /* 256 (may be the head's n)                  */
} engram_fwd;

/* THE forward pass of engram_expert.h, for position t -- the only one: inference and training both run it */
static void engram_expert_run(const engram_expert *e, const uint8_t *text, size_t n, size_t t, engram_fwd *c)
{
    const unsigned W = e->cfg.width, H = e->cfg.hidden;
    const double alpha = 1.0 / sqrt((double)W);
    unsigned j, v;
    engram_ctx_ids(&e->cfg.ctx, text, n, t, c->ids);
    /* the ids come from engram_ctx for this expert's own cfg: always inside the bag */
    (void)engram_cascade_gather(&e->bag, c->ids, e->cfg.ctx.orders, c->z[0]);
    c->sc[0] = 1.0 / sqrt((double)e->cfg.ctx.orders);
    c->r[0] = engram_norm_i(c->z[0], W, c->sc[0], c->n[0]);
    c->s[0] = engram_act_quant(c->n[0], e->gain, W, c->a[0], c->q[0]);
    for (j = 0; j < H; j++) {
        engram_cascade_matvec(&e->hid[j], c->q[j], c->z[j + 1u], 0u, W);
        c->sc[j + 1u] = c->s[j] * alpha;
        c->r[j + 1u] = engram_norm_i(c->z[j + 1u], W, c->sc[j + 1u], c->n[j + 1u]);
        c->s[j + 1u] = engram_act_quant(c->n[j + 1u], e->gain + (size_t)(j + 1u) * W, W, c->a[j + 1u], c->q[j + 1u]);
    }
    engram_cascade_matvec(&e->head, c->q[H], c->z[H + 1u], 0u, ENGRAM_EXPERT_VOCAB);
    c->sc[H + 1u] = c->s[H] * alpha;
    c->r[H + 1u] = engram_norm_i(c->z[H + 1u], ENGRAM_EXPERT_VOCAB, c->sc[H + 1u], c->n[H + 1u]);
    for (v = 0; v < ENGRAM_EXPERT_VOCAB; v++)
        c->logit[v] = c->n[H + 1u][v] * e->gain[(size_t)(H + 1u) * W + v] + e->bias[v];
}

/* the forward pass for position t, into e->logit, through the expert's one scratch */
static void engram_expert_forward(engram_expert *e, const uint8_t *text, size_t n, size_t t)
{
    engram_fwd c;
    unsigned j;
    for (j = 0; j <= e->cfg.hidden; j++) { c.z[j] = e->zi; c.n[j] = e->act; c.a[j] = e->act; c.q[j] = e->q; }
    c.z[e->cfg.hidden + 1u] = e->zi;
    c.n[e->cfg.hidden + 1u] = e->logit;
    c.logit = e->logit;
    engram_expert_run(e, text, n, t, &c);
}

/* -log2 p(target) from 256 logits, as engram_expert.h's bits(t); *m and *z: the max and the sum */
static double engram_expert_bits_of(const float *logit, unsigned target, double *mo, double *zo)
{
    static const double LOG2E = 1.44269504088896338700;
    double m = (double)logit[0], z = 0.0;
    unsigned v;
    for (v = 1; v < ENGRAM_EXPERT_VOCAB; v++) if ((double)logit[v] > m) m = (double)logit[v];
    for (v = 0; v < ENGRAM_EXPERT_VOCAB; v++) z += engram_exp((double)logit[v] - m);
    if (mo) *mo = m;
    if (zo) *zo = z;
    return engram_log2(z) - ((double)logit[target] - m) * LOG2E;
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
    size_t t;
    double total = 0.0;
    if (bits) *bits = 0.0;
    if (!e || !bits || (!text && n) || from > to || to > n) return ENGRAM_E_ARG;
    for (t = from; t < to; t++) {
        engram_expert_forward(e, text, n, t);
        total += engram_expert_bits_of(e->logit, text[t], NULL, NULL);
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

/* ==================================================================================================
 * THE BACKWARD PASS (engram_expert.h: THE BACKWARD PASS, EXACTLY)
 * ==================================================================================================
 * Phase A keeps everything each position computes in that position's own slot; phase B reduces over
 * the slots, each output element on its own, positions in order. Neither phase reads another's
 * position or another's element, so both split over threads with no change to a single bit (P2.2.3). */
struct engram_expert_grad {
    unsigned  W, hidden, orders, blog2;           /* the shape it was made for                     */
    size_t    bmax, B;
    size_t    lw, lz;                             /* per position: (hidden+1)*W, lw + 256          */
    /* phase A: position b's slot of each array starts at b * its width */
    uint32_t *ids;                                /* orders                                         */
    int32_t  *z;                                  /* lz: layer j at j*W, the head at lw             */
    float    *n;                                  /* lz                                             */
    float    *a;                                  /* lw                                             */
    int8_t   *q;                                  /* lw                                             */
    float    *x;                                  /* lw: (float)(q * s), the next matrix's input    */
    double   *sc, *r;                             /* hidden + 2                                     */
    double   *s;                                  /* hidden + 1                                     */
    float    *logit;                              /* 256                                            */
    float    *dl;                                 /* 256                                            */
    float    *dva;                                /* hidden*W + 256: hidden j-1's at (j-1)*W, the head's at hidden*W */
    float    *da;                                 /* lw                                             */
    float    *d0;                                 /* W                                              */
    double   *bits;                               /* 1                                              */
    uint64_t *keys;                               /* bmax * orders: (feature << 32) | (b * orders + k) */
    /* phase B: the gradients */
    float    *g_bias;                             /* 256                                            */
    float    *g_gain;                             /* lw + 256                                       */
    float    *g_head;                             /* 256 x W                                        */
    float    *g_hid[ENGRAM_EXPERT_MAX_HIDDEN];    /* W x W                                          */
    uint32_t *feat;                               /* bmax * orders at most                          */
    size_t    n_feat;
    float    *g_bag;                              /* n_feat x W                                     */
};

void engram_expert_grad_close(engram_expert_grad *g)
{
    unsigned j;
    if (!g) return;
    engram_free(g->ids); engram_free(g->z); engram_free(g->n); engram_free(g->a); engram_free(g->q);
    engram_free(g->x); engram_free(g->sc); engram_free(g->r); engram_free(g->s); engram_free(g->logit);
    engram_free(g->dl); engram_free(g->dva); engram_free(g->da); engram_free(g->d0); engram_free(g->bits);
    engram_free(g->keys); engram_free(g->g_bias); engram_free(g->g_gain); engram_free(g->g_head);
    for (j = 0; j < ENGRAM_EXPERT_MAX_HIDDEN; j++) engram_free(g->g_hid[j]);
    engram_free(g->feat); engram_free(g->g_bag);
    engram_free(g);
}

/* count x a x b elements of `size` bytes, or NULL (overflow or no memory) */
static void *engram_array3(size_t count, size_t a, size_t b, size_t size)
{
    if (a && count > SIZE_MAX / a) return NULL;
    count *= a;
    if (b && count > SIZE_MAX / b) return NULL;
    return engram_array(count * b, size);
}

engram_rc engram_expert_grad_open(engram_expert_grad **out, const engram_expert *e, size_t batch_max)
{
    engram_expert_grad *g;
    size_t M, W, H, O;
    unsigned j;
    int ok;
    if (!out) return ENGRAM_E_ARG;
    *out = NULL;
    if (!e || batch_max == 0u || batch_max > ENGRAM_EXPERT_MAX_BATCH) return ENGRAM_E_ARG;
    g = (engram_expert_grad *)engram_calloc(1u, sizeof *g);
    if (!g) return ENGRAM_E_MEM;
    g->W = e->cfg.width; g->hidden = e->cfg.hidden; g->orders = e->cfg.ctx.orders; g->blog2 = e->cfg.ctx.buckets_log2;
    g->bmax = M = batch_max;
    W = g->W; H = g->hidden; O = g->orders;
    g->lw = (H + 1u) * W;
    g->lz = g->lw + ENGRAM_EXPERT_VOCAB;
    g->ids = (uint32_t *)engram_array3(M, O, 1u, sizeof *g->ids);
    g->z = (int32_t *)engram_array3(M, g->lz, 1u, sizeof *g->z);
    g->n = (float *)engram_array3(M, g->lz, 1u, sizeof *g->n);
    g->a = (float *)engram_array3(M, g->lw, 1u, sizeof *g->a);
    g->q = (int8_t *)engram_array3(M, g->lw, 1u, sizeof *g->q);
    g->x = (float *)engram_array3(M, g->lw, 1u, sizeof *g->x);
    g->sc = (double *)engram_array3(M, H + 2u, 1u, sizeof *g->sc);
    g->r = (double *)engram_array3(M, H + 2u, 1u, sizeof *g->r);
    g->s = (double *)engram_array3(M, H + 1u, 1u, sizeof *g->s);
    g->logit = (float *)engram_array3(M, ENGRAM_EXPERT_VOCAB, 1u, sizeof *g->logit);
    g->dl = (float *)engram_array3(M, ENGRAM_EXPERT_VOCAB, 1u, sizeof *g->dl);
    g->dva = (float *)engram_array3(M, H * W + ENGRAM_EXPERT_VOCAB, 1u, sizeof *g->dva);
    g->da = (float *)engram_array3(M, g->lw, 1u, sizeof *g->da);
    g->d0 = (float *)engram_array3(M, W, 1u, sizeof *g->d0);
    g->bits = (double *)engram_array3(M, 1u, 1u, sizeof *g->bits);
    g->keys = (uint64_t *)engram_array3(M, O, 1u, sizeof *g->keys);
    g->g_bias = (float *)engram_array(ENGRAM_EXPERT_VOCAB, sizeof *g->g_bias);
    g->g_gain = (float *)engram_array(g->lz, sizeof *g->g_gain);
    g->g_head = (float *)engram_array3(ENGRAM_EXPERT_VOCAB, W, 1u, sizeof *g->g_head);
    g->feat = (uint32_t *)engram_array3(M, O, 1u, sizeof *g->feat);
    g->g_bag = (float *)engram_array3(M, O, W, sizeof *g->g_bag);
    ok = g->ids && g->z && g->n && g->a && g->q && g->x && g->sc && g->r && g->s && g->logit && g->dl &&
         g->dva && g->da && g->d0 && g->bits && g->keys && g->g_bias && g->g_gain && g->g_head && g->feat && g->g_bag;
    for (j = 0; ok && j < H; j++) ok = (g->g_hid[j] = (float *)engram_array3(W, W, 1u, sizeof(float))) != NULL;
    if (!ok) { engram_expert_grad_close(g); return ENGRAM_E_MEM; }
    *out = g;
    return ENGRAM_OK;
}

/* norm_back of engram_expert.h: dv from dy, into out (which may be dy) */
static void engram_norm_back(const float *dy, const float *gain, const int32_t *z, double sc, double r, unsigned n,
                             float *out)
{
    double suv = 0.0, r3 = r * r * r;
    unsigned i;
    for (i = 0; i < n; i++) suv += ((double)dy[i] * (double)gain[i]) * ((double)z[i] * sc);
    for (i = 0; i < n; i++) {
        double u = (double)dy[i] * (double)gain[i];
        out[i] = (float)(u / r - ((double)z[i] * sc) * suv / ((double)n * r3));
    }
}

/* PHASE A for position b of the batch: forward with every layer kept, then back to the bag */
static void engram_expert_back_pos(const engram_expert *e, engram_expert_grad *g, const uint8_t *text, size_t n,
                                   size_t t, size_t b)
{
    const unsigned W = g->W, H = g->hidden, V = ENGRAM_EXPERT_VOCAB;
    const double alpha = 1.0 / sqrt((double)W);
    const double bd = (double)g->B;
    engram_fwd c;
    double m, Z;
    float *dl = g->dl + b * V, *dvh = g->dva + b * (H * W + V) + (size_t)H * W, *da;
    unsigned j, v, i;
    for (j = 0; j <= H; j++) {
        c.z[j] = g->z + b * g->lz + (size_t)j * W;
        c.n[j] = g->n + b * g->lz + (size_t)j * W;
        c.a[j] = g->a + b * g->lw + (size_t)j * W;
        c.q[j] = g->q + b * g->lw + (size_t)j * W;
    }
    c.z[H + 1u] = g->z + b * g->lz + g->lw;
    c.n[H + 1u] = g->n + b * g->lz + g->lw;
    c.logit = g->logit + b * V;
    engram_expert_run(e, text, n, t, &c);
    memcpy(g->ids + b * g->orders, c.ids, g->orders * sizeof *c.ids);
    memcpy(g->sc + b * (H + 2u), c.sc, (H + 2u) * sizeof *c.sc);
    memcpy(g->r + b * (H + 2u), c.r, (H + 2u) * sizeof *c.r);
    memcpy(g->s + b * (H + 1u), c.s, (H + 1u) * sizeof *c.s);
    for (j = 0; j <= H; j++) {
        float *x = g->x + b * g->lw + (size_t)j * W;
        for (i = 0; i < W; i++) x[i] = (float)((double)c.q[j][i] * c.s[j]);
    }
    /* the loss and dL/dlogit */
    g->bits[b] = engram_expert_bits_of(c.logit, text[t], &m, &Z);
    for (v = 0; v < V; v++) {
        double p = engram_exp((double)c.logit[v] - m) / Z;
        dl[v] = (float)((p - (v == text[t] ? 1.0 : 0.0)) / bd);
    }
    /* the head */
    engram_norm_back(dl, e->gain + (size_t)(H + 1u) * W, c.z[H + 1u], c.sc[H + 1u], c.r[H + 1u], V, dvh);
    for (v = 0; v < V; v++) dvh[v] = (float)((double)dvh[v] * alpha);
    da = g->da + b * g->lw + (size_t)H * W;
    engram_cascade_matvec_t(&e->head, dvh, da, 0u, W);
    /* the layers, top down */
    for (j = H + 1u; j-- > 0u;) {
        float *dv = j > 0u ? g->dva + b * (H * W + V) + (size_t)(j - 1u) * W : g->d0 + b * W;
        for (i = 0; i < W; i++) if (!(c.a[j][i] > 0.0f)) da[i] = 0.0f;
        engram_norm_back(da, e->gain + (size_t)j * W, c.z[j], c.sc[j], c.r[j], W, dv);
        if (j > 0u) {
            for (i = 0; i < W; i++) dv[i] = (float)((double)dv[i] * alpha);
            da = g->da + b * g->lw + (size_t)(j - 1u) * W;
            engram_cascade_matvec_t(&e->hid[j - 1u], dv, da, 0u, W);
        } else {
            for (i = 0; i < W; i++) dv[i] = (float)((double)dv[i] * c.sc[0]);
        }
    }
}

/* PHASE B: the float units -- bias, then every gain -- elements [k0, k1) of bias (0 .. 256) ++ gain */
static void engram_expert_reduce_units(engram_expert_grad *g, size_t k0, size_t k1)
{
    const size_t V = ENGRAM_EXPERT_VOCAB;
    size_t k, b;
    for (k = k0; k < k1; k++) {
        float acc = 0.0f;
        if (k < V) {
            for (b = 0; b < g->B; b++) acc += g->dl[b * V + k];
            g->g_bias[k] = acc;
        } else if (k - V < g->lw) {                                      /* a layer's gain       */
            size_t i = k - V;
            for (b = 0; b < g->B; b++) acc += g->da[b * g->lw + i] * g->n[b * g->lz + i];
            g->g_gain[i] = acc;
        } else {                                                         /* the head's gain      */
            size_t v = k - V - g->lw;
            for (b = 0; b < g->B; b++) acc += g->dl[b * V + v] * g->n[b * g->lz + g->lw + v];
            g->g_gain[g->lw + v] = acc;
        }
    }
}

/* PHASE B: rows [r0, r1) of matrix `which` (0 .. hidden-1: hidden layer; hidden: the head) */
static void engram_expert_reduce_matrix(engram_expert_grad *g, unsigned which, size_t r0, size_t r1)
{
    const size_t W = g->W, stride = (size_t)g->hidden * W + ENGRAM_EXPERT_VOCAB;
    float *G = which == g->hidden ? g->g_head : g->g_hid[which];
    size_t r, b, c;
    /* the head's input is the last layer's x; hidden layer `which` reads layer which's x */
    const size_t xoff = (size_t)which * W;
    for (r = r0; r < r1; r++) {
        float *row = G + r * W;
        for (c = 0; c < W; c++) row[c] = 0.0f;
        for (b = 0; b < g->B; b++) {
            const float d = g->dva[b * stride + (size_t)which * W + r];
            const float *x = g->x + b * g->lw + xoff;
            for (c = 0; c < W; c++) row[c] += d * x[c];
        }
    }
}

/* ascending uint64 keys, in place, no memory: heap sort (the keys are distinct, so any sort gives this) */
static void engram_sort_u64(uint64_t *k, size_t n)
{
    size_t i, end;
    if (n < 2u) return;
    for (i = n / 2u; i-- > 0u;) {
        size_t root = i;
        for (;;) {
            size_t ch = 2u * root + 1u;
            uint64_t t;
            if (ch >= n) break;
            if (ch + 1u < n && k[ch + 1u] > k[ch]) ch++;
            if (k[root] >= k[ch]) break;
            t = k[root]; k[root] = k[ch]; k[ch] = t;
            root = ch;
        }
    }
    for (end = n - 1u; end > 0u; end--) {
        size_t root = 0u;
        uint64_t t = k[0]; k[0] = k[end]; k[end] = t;
        for (;;) {
            size_t ch = 2u * root + 1u;
            if (ch >= end) break;
            if (ch + 1u < end && k[ch + 1u] > k[ch]) ch++;
            if (k[root] >= k[ch]) break;
            t = k[root]; k[root] = k[ch]; k[ch] = t;
            root = ch;
        }
    }
}

/* PHASE B: the bag. Every (position, order) as one key -- feature, then position, then order -- sorted:
 * each feature's occurrences are then consecutive and in the order the specification sums them. */
static void engram_expert_reduce_bag(engram_expert_grad *g)
{
    const size_t W = g->W, O = g->orders, nk = g->B * O;
    size_t i, c;
    for (i = 0; i < nk; i++) g->keys[i] = ((uint64_t)g->ids[i] << 32) | (uint64_t)i;
    engram_sort_u64(g->keys, nk);
    g->n_feat = 0u;
    for (i = 0; i < nk; i++) {
        uint32_t f = (uint32_t)(g->keys[i] >> 32);
        const float *d0 = g->d0 + ((size_t)(uint32_t)g->keys[i] / O) * W;
        float *row;
        if (g->n_feat == 0u || g->feat[g->n_feat - 1u] != f) {
            row = g->g_bag + g->n_feat * W;
            for (c = 0; c < W; c++) row[c] = 0.0f;
            g->feat[g->n_feat++] = f;
        }
        row = g->g_bag + (g->n_feat - 1u) * W;
        for (c = 0; c < W; c++) row[c] += d0[c];
    }
}

engram_rc engram_expert_backward(const engram_expert *e, engram_expert_grad *g, const uint8_t *text, size_t n,
                                 const size_t *pos, size_t B, double *bits)
{
    size_t b;
    double total = 0.0;
    unsigned j;
    if (bits) *bits = 0.0;
    if (!e || !g || !text || !pos || !bits || B == 0u || B > g->bmax) return ENGRAM_E_ARG;
    if (g->W != e->cfg.width || g->hidden != e->cfg.hidden || g->orders != e->cfg.ctx.orders ||
        g->blog2 != e->cfg.ctx.buckets_log2)
        return ENGRAM_E_ARG;
    for (b = 0; b < B; b++) if (pos[b] >= n) return ENGRAM_E_ARG;
    g->B = B;
    for (b = 0; b < B; b++) engram_expert_back_pos(e, g, text, n, pos[b], b);
    for (b = 0; b < B; b++) total += g->bits[b];
    engram_expert_reduce_units(g, 0u, ENGRAM_EXPERT_VOCAB + g->lz);
    for (j = 0; j <= g->hidden; j++)
        engram_expert_reduce_matrix(g, j, 0u, j == g->hidden ? ENGRAM_EXPERT_VOCAB : g->W);
    engram_expert_reduce_bag(g);
    *bits = total;
    return ENGRAM_OK;
}

void engram_expert_grad_get(const engram_expert_grad *g, engram_expert_grad_view *v)
{
    unsigned j;
    if (!v) return;
    memset(v, 0, sizeof *v);
    if (!g) return;
    v->bias = g->g_bias;
    v->gain = g->g_gain;
    v->n_gain = g->lz;
    v->head = g->g_head;
    for (j = 0; j < g->hidden; j++) v->hidden[j] = g->g_hid[j];
    v->feat = g->feat;
    v->n_feat = g->n_feat;
    v->bag = g->g_bag;
}
