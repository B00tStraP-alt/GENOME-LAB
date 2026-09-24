#!/usr/bin/env python3
"""
mutants_p21.py -- the P2.1 mutation campaign: is every step of the deterministic math, the Cascade
weight, the hashed contexts and the expert's forward pass actually TESTED? The runner is tools/mutants.py.

Usage:  tools/mutants_p21.py [prefix ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutants  # noqa: E402

M = [
 # ---- engram_math
 ("A1 exp: a Taylor coefficient dropped", "engram_math.c", "    p = p * r + 0.5;", "    p = p * r + 0.5000001;"),
 ("A2 exp: the low half of ln 2 lost", "engram_math.c", "LN2_LO = 1.90821492927058770002e-10;", "LN2_LO = 0.0;"),
 ("A3 exp: subnormal scaling in one step", "engram_math.c", "    if (k < -1021) return (p * engram_pow2i(k + 1000)) * engram_pow2i(-1000);", "    if (k < -1075) return 0.0;"),
 ("A4 round: halves toward zero", "engram_math.c", "    if (t - r >= 0.5) r += 1.0;", "    if (t - r > 0.5) r += 1.0;"),
 ("A5 log2: no range reduction to sqrt(2)", "engram_math.c", "    if (m > 1.4142135623730951) { m *= 0.5; e++; }", "    (void)0;"),
 ("A6 log2: a series term wrong", "engram_math.c", "    t = t * s2 + 1.0 / 3.0;", "    t = t * s2 + 1.0 / 3.0001;"),
 ("A7 pow2i: exponent bias off by one", "engram_math.c", "(uint64_t)(k + 1023) << 52", "(uint64_t)(k + 1022) << 52"),
 # ---- engram_cascade
 ("B1 cascade: no cap of one basin a step", "engram_cascade.c", "    if (mag > (float)K) mag = (float)K;", "    (void)0;"),
 ("B2 cascade: metaplasticity never applied", "engram_cascade.c", "    if ((v > 0 && !up) || (v < 0 && up)) mag *= m->meta[engram_cascade_depth(p, K)];", "    (void)0;"),
 ("B3 cascade: metaplasticity on moves AWAY from zero", "engram_cascade.c", "    if ((v > 0 && !up) || (v < 0 && up)) mag *= m->meta[engram_cascade_depth(p, K)];", "    if ((v > 0 && up) || (v < 0 && !up)) mag *= m->meta[engram_cascade_depth(p, K)];"),
 ("B4 cascade: biased rounding (always the floor)", "engram_cascade.c", "(engram_cascade_uniform(m->seed, step, key) < mag - whole ? 1u : 0u)", "((void)step, (void)key, 0u)"),
 ("B5 cascade: no clamp at the top", "engram_cascade.c", "    if (up) p = p + steps > top ? top : p + steps;", "    (void)top; if (up) p = p + steps > 255u ? 255u : p + steps;"),
 ("B6 cascade: flips not counted", "engram_cascade.c", "        if (nv != v) (*fl)++;", "        (void)fl;"),
 ("B7 cascade: value plane not kept in step", "engram_cascade.c", "        m->val[cell] = (int8_t)nv;", "        (void)0;"),
 ("B8 cascade: row normalised by the wrong count", "engram_cascade.c", "ss / (double)cols", "ss / (double)(cols + 1u)"),
 ("B9 cascade: the unit update normalised per touched feature", "engram_cascade.c", "ss / (double)F", "ss / (double)n_feat"),
 ("B10 cascade: the unit update keyed by the stored index", "engram_cascade.c", "(uint64_t)h * F + feat[i], &mv, &fl);", "(uint64_t)feat[i] * H + h, &mv, &fl);"),
 ("B11 cascade: the draw keyed by the step only", "engram_cascade.c", "    uint64_t base = engram_mix64(seed * 0x9E3779B97F4A7C15ull + step);", "    uint64_t base = engram_mix64(step); (void)seed;"),
 ("B12 cascade: the transposed product skips negative weights", "engram_cascade.c", "            else if (v[c] < 0) y[c] -= dr;", "            else if (v[c] < 0) y[c] -= 0.0f;"),
 ("B13 cascade: gather accepts an id outside the matrix", "engram_cascade.c", "    for (i = 0; i < n; i++) if (rows[i] >= m->rows) return ENGRAM_E_ARG;", "    (void)0;"),
 ("B14 cascade: a loaded position of 3K accepted", "engram_cascade.c", "if (p[36u + i] >= 3u * K) return ENGRAM_E_FORMAT;", "if (p[36u + i] > 3u * K) return ENGRAM_E_FORMAT;"),
 ("B15 cascade: metaplastic table exponent halved", "engram_cascade.c", "(double)d * 4.0 / (double)K", "(double)d * 2.0 / (double)K"),
 # ---- engram_ctx
 ("X1 ctx: before-the-start is a zero byte", "engram_ctx.c", "t >= k ? (uint64_t)text[t - k] : 256u;", "t >= k ? (uint64_t)text[t - k] : 0u;"),
 ("X2 ctx: orders share one block", "engram_ctx.c", "(((uint64_t)(k - 1u) << c->buckets_log2) | (h & mask))", "((uint64_t)0u | (h & mask))"),
 ("X3 ctx: order k hashes only its own byte", "engram_ctx.c", "        h = engram_mix2(h, b);", "        h = engram_mix2(c->seed + k, b);"),
 # ---- engram_expert
 ("E1 expert: the bag's 1/sqrt(orders) scale dropped", "engram_expert.c", "c->sc[0] = 1.0 / sqrt((double)e->cfg.ctx.orders);", "c->sc[0] = 1.0;"),
 ("E2 expert: the norm's epsilon changed", "engram_expert.c", "r = sqrt(ss / (double)n + 1e-6);", "r = sqrt(ss / (double)n + 1e-5);"),
 ("E3 expert: no ReLU", "engram_expert.c", "        a[i] = y > 0.0f ? y : 0.0f;", "        a[i] = y;"),
 ("E4 expert: int8 scale 128", "engram_expert.c", "s = (double)mx / 127.0 + 1e-12;", "s = (double)mx / 128.0 + 1e-12;"),
 ("E5 expert: the head's bias ignored", "engram_expert.c", " + e->bias[v];", ";"),
 ("E6 expert: the max logit not found (first one kept)", "engram_expert.c", "if ((double)logit[v] > m) m = (double)logit[v];", "if (0) m = (double)logit[v];"),
 ("E7 expert: the head seeded like the bag", "engram_expert.c", "engram_mix2(cfg->seed, 99u)", "engram_mix2(cfg->seed, 1u)"),
 ("E8 expert: the size bound not enforced", "engram_expert.c", "if (engram_expert_weights(c) > ENGRAM_EXPERT_MAX_WEIGHTS) return ENGRAM_E_FULL;", "(void)0;"),
 ("E9 expert: hidden layers read the first layer's gains", "engram_expert.c", "engram_act_quant(c->n[j + 1u], e->gain + (size_t)(j + 1u) * W, W,", "engram_act_quant(c->n[j + 1u], e->gain, W,"),
]

if __name__ == "__main__":
    mutants.run(M, ["test_math", "test_cascade", "test_expert"])
