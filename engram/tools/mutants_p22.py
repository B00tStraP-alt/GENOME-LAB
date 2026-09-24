#!/usr/bin/env python3
"""
mutants_p22.py -- the P2.2 mutation campaign: is every step of the backward pass actually TESTED? The
runner is tools/mutants.py.

Usage:  tools/mutants_p22.py [prefix ...]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutants  # noqa: E402

F = "engram_expert.c"
M = [
 # ---- phase A: the loss and dL/dlogit
 ("G1 backward: dl not divided by the batch", F, "dl[v] = (float)((p - (v == text[t] ? 1.0 : 0.0)) / bd);", "dl[v] = (float)((p - (v == text[t] ? 1.0 : 0.0)) / (bd * 0.0 + 1.0));"),
 ("G2 backward: the target never subtracted", F, "dl[v] = (float)((p - (v == text[t] ? 1.0 : 0.0)) / bd);", "dl[v] = (float)(p / bd);"),
 ("G3 backward: softmax normalised by a slightly wrong sum", F, "double p = engram_exp((double)c.logit[v] - m) / Z;", "double p = engram_exp((double)c.logit[v] - m) / (Z * 1.0000001);"),
 ("G4 backward: the batch's loss misses its last position", F, "for (b = 0; b < B; b++) total += g->bits[b];", "for (b = 0; b + 1u < B; b++) total += g->bits[b];"),
 # ---- the norm
 ("G5 norm_back: the mean's correction term dropped", F, "out[i] = (float)(u / r - ((double)z[i] * sc) * suv / ((double)n * r3));", "out[i] = (float)(u / r - 0.0 * suv * r3);"),
 ("G6 norm_back: r squared, not cubed", F, "double suv = 0.0, r3 = r * r * r;", "double suv = 0.0, r3 = r * r;"),
 ("G7 norm_back: the gain not applied to the gradient", F, "        double u = (double)dy[i] * (double)gain[i];", "        double u = (double)dy[i];"),
 ("G8 norm_back: the correction summed without the gain", F, "suv += ((double)dy[i] * (double)gain[i]) * ((double)z[i] * sc);", "suv += (double)dy[i] * ((double)z[i] * sc);"),
 # ---- the layers
 ("G9 head: dva not scaled by 1/sqrt(W)", F, "for (v = 0; v < V; v++) dvh[v] = (float)((double)dvh[v] * alpha);", "(void)0;"),
 ("G10 layers: no ReLU mask", F, "if (!(c.a[j][i] > 0.0f)) da[i] = 0.0f;", "(void)0;"),
 ("G11 layers: the mask read before the gain (wrong where a gain is negative)", F, "if (!(c.a[j][i] > 0.0f)) da[i] = 0.0f;", "if (!(c.n[j][i] > 0.0f)) da[i] = 0.0f;"),
 ("G12 hidden: dva not scaled by 1/sqrt(W)", F, "for (i = 0; i < W; i++) dv[i] = (float)((double)dv[i] * alpha);", "(void)0;"),
 ("G13 bag: d0 not scaled by 1/sqrt(ORDERS)", F, "for (i = 0; i < W; i++) dv[i] = (float)((double)dv[i] * c.sc[0]);", "(void)0;"),
 ("G14 layers: the input taken as q, not q * s", F, "x[i] = (float)((double)c.q[j][i] * c.s[j]);", "x[i] = (float)c.q[j][i];"),
 ("G15 layers: every hidden layer back through the first one", F, "engram_cascade_matvec_t(&e->hid[j - 1u], dv, da, 0u, W);", "engram_cascade_matvec_t(&e->hid[0], dv, da, 0u, W);"),
 ("G16 layers: every layer's norm with the first layer's gains", F, "engram_norm_back(da, e->gain + (size_t)j * W,", "engram_norm_back(da, e->gain,"),
 # ---- phase B: the reductions
 ("G17 units: a layer's gain gradient without the norm output", F, "acc += g->da[b * g->lw + i] * g->n[b * g->lz + i];", "acc += g->da[b * g->lw + i];"),
 ("G18 units: the head's gain gradient without the norm output", F, "acc += g->dl[b * V + v] * g->n[b * g->lz + g->lw + v];", "acc += g->dl[b * V + v];"),
 ("G19 matrices: every matrix reads the first layer's input", F, "const size_t xoff = (size_t)which * W;", "const size_t xoff = 0u;"),
 ("G20 matrices: rows not cleared (accumulated across calls)", F,
  "        for (c = 0; c < W; c++) row[c] = 0.0f;\n        for (b = 0; b < g->B; b++) {", "        for (b = 0; b < g->B; b++) {"),
 ("G21 bag: occurrences of one feature not merged", F, "if (g->n_feat == 0u || g->feat[g->n_feat - 1u] != f) {", "if (1) {"),
 ("G22 bag: the keys not sorted", F, "    engram_sort_u64(g->keys, nk);", "    engram_sort_u64(g->keys, 0u);"),
 ("G23 sort: the heap built toward the smaller child", F, "            if (ch + 1u < n && k[ch + 1u] > k[ch]) ch++;", "            if (ch + 1u < n && k[ch + 1u] < k[ch]) ch++;"),
 ("G24 bag: the gradient of the wrong position", F, "((size_t)(uint32_t)g->keys[i] / O) * W", "((size_t)(uint32_t)g->keys[i] % O) * W"),
 # ---- the contract
 ("G25 contract: a position past the text accepted", F, "for (b = 0; b < B; b++) if (pos[b] >= n) return ENGRAM_E_ARG;", "(void)0;"),
 ("G26 contract: a workspace of another shape accepted", F, "if (g->W != e->cfg.width || g->hidden != e->cfg.hidden", "if (0"),
 ("G27 contract: one position more than the workspace holds", F, "B == 0u || B > g->bmax)", "B == 0u || B > g->bmax + 1u)"),
 ("G28 contract: the batch cap not enforced", F, "batch_max > ENGRAM_EXPERT_MAX_BATCH)", "batch_max > ENGRAM_EXPERT_MAX_BATCH + 1u)"),
 ("G29 open: a hidden gradient's allocation failure ignored", F,
  "for (j = 0; ok && j < H; j++) ok = (g->g_hid[j] = (float *)engram_array3(W, W, 1u, sizeof(float))) != NULL;",
  "for (j = 0; ok && j < H; j++) g->g_hid[j] = (float *)engram_array3(W, W, 1u, sizeof(float));"),
 ("G30 close: the hidden gradients leaked", F, "for (j = 0; j < ENGRAM_EXPERT_MAX_HIDDEN; j++) engram_free(g->g_hid[j]);", "(void)j;"),
]

if __name__ == "__main__":
    mutants.run(M, ["test_learn"])
