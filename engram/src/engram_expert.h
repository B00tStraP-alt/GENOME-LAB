/* ==================================================================================================
 * engram_expert.h -- ONE EXPERT of the slow store: a next-byte predictor in Cascade weights.
 * ==================================================================================================
 *
 * THE NETWORK (research/p2_slow: every choice below was measured, pre-registered, and confirmed once)
 * ==================================================================================================
 *      the ORDERS hashed contexts before the cursor                             engram_ctx.h
 *        -> BAG      the sum of their W-wide rows of a Cascade table (F x W)      integer, exact
 *        -> RMS norm (scale 1/sqrt(ORDERS)), gain, ReLU, int8
 *        -> HIDDEN   0 .. 4 W x W Cascade layers: matvec (exact), RMS norm, gain, ReLU, int8
 *        -> HEAD     a 256 x W Cascade layer: matvec, RMS norm, gain, bias         256 logits
 *        -> softmax  with engram_math.h's exp and log2
 *
 * The default is what P2.1.1 chose: 6 orders x 4096 buckets, width 512, no hidden layer, Cascade-16
 * with kappa 0.5 -- 12.7 M weights, one byte each -- and 1,024 floats (the norms' gains, the head's
 * bias). On held-out English it assigns 2.05 bits per byte after training, against 2.19 for the best
 * count model and 2.00 for the same network in float32 weights (research/p2_slow/README.md).
 *
 * THE FORWARD PASS, EXACTLY (so an independent implementation can agree to the bit -- tools/
 * gen_expert_vectors.py does):
 *   norm(z, scale):  v_i = (double)z_i * scale;  r = sqrt(sum_i v_i^2 / n + 1e-6), summed in index order
 *                    in double;  out_i = (float)(v_i / r)
 *   act(a, gain):    y_i = a_i * gain_i (float);  a_i = max(y_i, 0);  s = (double)max_i a_i / 127 + 1e-12;
 *                    q_i = round_half_away((double)a_i / s)
 *   bag:             z = sum of the value rows of the ids;  a = norm(z, 1/sqrt(ORDERS));  q, s = act(a, g0)
 *   each hidden j:   z = W_j q;  a = norm(z, s / sqrt(W));  q, s = act(a, g_{j+1})
 *   head:            z = W_head q;  l_v = norm(z, s / sqrt(W))_v * g_head,v + bias_v   (float)
 *   bits(t):         m = max_v l_v;  Z = sum_v exp(l_v - m) in index order (double);
 *                    bits = log2(Z) - (l_target - m) * log2(e)
 * Every float operation is + - * / or sqrt or engram_math, in a fixed order, under -ffp-contract=off;
 * the products are integers. So the bits are the same on every platform (EXPERTPRINT, compared by the
 * gate).
 * ============================================================================================== */
#ifndef ENGRAM_EXPERT_H
#define ENGRAM_EXPERT_H

#include "engram.h"
#include "engram_cascade.h"
#include "engram_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_EXPERT_MAX_HIDDEN 4u
#define ENGRAM_EXPERT_VOCAB      256u
/* The largest expert open() will build: 256 M weights (a quarter of a gigabyte at one byte each). */
#define ENGRAM_EXPERT_MAX_WEIGHTS ((size_t)1 << 28)

typedef struct {
    engram_ctx_cfg ctx;
    unsigned       width;          /* W: 16 .. 4096, a multiple of 16                               */
    unsigned       hidden;         /* W x W layers between the bag and the head, 0 .. MAX_HIDDEN    */
    unsigned       K;              /* Cascade positions per basin: 4, 16 or 64                      */
    float          kappa;          /* metaplasticity, 0 .. 16                                       */
    float          lr;             /* Cascade learning rate (P2.2), positions per normalised unit   */
    float          lr_unit;        /* Adam learning rate for the gains and the bias (P2.2)          */
    uint64_t       seed;
} engram_expert_cfg;

void      engram_expert_cfg_default(engram_expert_cfg *c);
/* ENGRAM_E_ARG for anything out of range; ENGRAM_E_FULL if the expert would exceed MAX_WEIGHTS. */
engram_rc engram_expert_cfg_check(const engram_expert_cfg *c);

typedef struct engram_expert engram_expert;

engram_rc engram_expert_open(engram_expert **out, const engram_expert_cfg *cfg);   /* cfg NULL: defaults */
void      engram_expert_close(engram_expert *e);
const engram_expert_cfg *engram_expert_cfg_get(const engram_expert *e);

/* The 256 logits of the byte at position t of text[0 .. n) (t == n: the byte after the text). */
engram_rc engram_expert_logits(engram_expert *e, const uint8_t *text, size_t n, size_t t, float logits[256]);

/* Bits the expert assigns to text[from .. to) given everything before each byte:
 *   sum over t of -log2 p(text[t] | the ORDERS bytes before t).
 * ENGRAM_E_ARG for from > to or to > n. The engine of every Phase 2 measurement and the Phase 3
 * verdict. Uses the expert's scratch: one thread per expert at a time. */
engram_rc engram_expert_bits(engram_expert *e, const uint8_t *text, size_t n, size_t from, size_t to,
                             double *bits);

/* The float parameters, in place: gains ((hidden + 1) * W for the norms before the head, then 256 for
 * the head's norm) and the head's 256 biases. For training (P2.2), for snapshots (P3.1), and for tests
 * that must set them to known values. */
typedef struct {
    float  *gain;
    size_t  n_gain;
    float  *bias;
    size_t  n_bias;
} engram_expert_units;

void engram_expert_units_get(engram_expert *e, engram_expert_units *u);

typedef struct {
    size_t   weights;          /* Cascade weights, all matrices                                    */
    size_t   units;            /* float parameters                                                 */
    size_t   bytes_resident;   /* everything the expert holds                                      */
} engram_expert_stats;

void      engram_expert_stats_get(const engram_expert *e, engram_expert_stats *st);

/* A 64-bit hash of the configuration, every Cascade matrix (positions, step) and every float
 * parameter's bits. */
uint64_t  engram_expert_fingerprint(const engram_expert *e);

/* ---- THE BACKWARD PASS, EXACTLY ------------------------------------------------------------------
 * Over a batch of B positions, in the order given, the gradient of the batch's MEAN loss in nats,
 * (1/B) sum_b -ln p(target_b) -- which is bits * ln 2 / B of the bits returned. Straight-through: the int8
 * quantisation is treated as the identity, so a layer's input x is taken as x_c = (float)((double)q_c * s).
 *
 * PHASE A, per position (independent of every other position):
 *   p_v = exp(l_v - m) / Z  (double, Z summed in index order);  dl_v = (float)((p_v - [v == target]) / B)
 *   norm_back(dy, gain, z, scale, r):  u_i = (double)dy_i * gain_i;  suv = sum_i u_i * ((double)z_i * scale)
 *                                      (double, index order);  dv_i = (float)(u_i / r - ((double)z_i * scale)
 *                                      * suv / (n r^3)),  r^3 = r * r * r
 *   (z, scale, r: the layer's own, from its forward; alpha = 1 / sqrt(W) and c0 = 1 / sqrt(ORDERS), the
 *   doubles the forward pass scales by)
 *   head:     dv = norm_back(dl, head gains, ...);  dva_v = (float)((double)dv_v * alpha)
 *             dx = the transposed product of the head's values with dva (engram_cascade_matvec_t)
 *   layer j = hidden .. 0:  da_i = a_i > 0 ? dx_i : 0 (a: after gain and ReLU);  dv = norm_back(da, layer
 *             j's gains, ...)
 *             j > 0: dva_i = (float)((double)dv_i * alpha);  dx = transposed product of hidden j-1 with dva
 *             j = 0: d0_i = (float)((double)dv_i * c0)
 * PHASE B, reductions, each output element from +0, summed over the positions IN ORDER (float: the
 * product rounded, then the sum):
 *   bias_v   += dl_v                       head gain_v  += dl_v * n_head,v      layer gain_i += da_i * n_i
 *   head W[v][c] += dva_v * x_c (x: the last layer's)      hidden j W[i][c] += dva_i * x_c (x: layer j-1's)
 *   bag  row f, unit i  += d0_i  for every order k whose id is f (positions in order, then orders)
 * So a split of the positions, or of the output elements, over threads changes nothing (P2.2.3). The
 * loss returned is the batch's bits, summed in position order as engram_expert_bits sums them.
 *
 * The FORMULAS are proven against central finite differences of the loss in a float64 twin of the network
 * (tools/ref_expert.py `shadow_gradcheck`, which also proves it can fail); the C is proven equal, bit for
 * bit, to the Python implementation of the text above (test_expert). */
typedef struct engram_expert_grad engram_expert_grad;

#define ENGRAM_EXPERT_MAX_BATCH 65536u

/* A gradient workspace for batches of up to batch_max (1 .. MAX_BATCH) positions, for experts of e's shape. */
engram_rc engram_expert_grad_open(engram_expert_grad **out, const engram_expert *e, size_t batch_max);
void      engram_expert_grad_close(engram_expert_grad *g);

/* Forward and backward over text[pos[0]], ..., text[pos[B-1]] (each < n); the workspace's gradients are
 * REPLACED, not accumulated. *bits: the batch's loss. ENGRAM_E_ARG for B == 0, B > batch_max, a position
 * >= n, or a workspace made for another expert's shape. Allocates nothing. */
engram_rc engram_expert_backward(const engram_expert *e, engram_expert_grad *g, const uint8_t *text, size_t n,
                                 const size_t *pos, size_t B, double *bits);

typedef struct {
    const float    *bias;            /* 256                                              */
    const float    *gain;            /* n_gain, laid out as engram_expert_units' gain    */
    size_t          n_gain;
    const float    *head;            /* 256 x W                                          */
    const float    *hidden[ENGRAM_EXPERT_MAX_HIDDEN];   /* W x W each                    */
    const uint32_t *feat;            /* the batch's distinct features, ascending         */
    size_t          n_feat;
    const float    *bag;             /* n_feat x W: row i is feature feat[i]'s gradient  */
} engram_expert_grad_view;

void      engram_expert_grad_get(const engram_expert_grad *g, engram_expert_grad_view *v);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_EXPERT_H */
