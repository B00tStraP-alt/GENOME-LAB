#!/usr/bin/env python3
"""
gen_expert_vectors.py -- freeze what the expert's forward pass (engram_expert.c) must compute, from an
independent Python implementation of the pass as engram_expert.h specifies it: every float32 operation in
numpy float32, every double one in Python floats (IEEE doubles), the integer products in int64, exp and
log2 restated operation for operation (Python floats ARE doubles, so the same operations give the same
bits), the Cascade initialisation and the context hashes from their own specifications.

Per case: the configuration, the gains and biases (set to a fixed pattern of exact binary fractions, so
the gain and bias arithmetic is exercised, not multiplied by 1 and added to 0), and for a text:
the bits of every position, and the logits of three positions.

Usage: tools/gen_expert_vectors.py test/expert_vectors.h      (python3 with numpy; ~2 minutes -- the
       default configuration's 12.7 M positions are drawn in pure Python)
"""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_cascade_vectors import mix64, value, exp_poly, scale, eround, pow2i  # noqa: E402

F32 = np.float32
M64 = (1 << 64) - 1
LOG2E = 1.44269504088896338700


def mix2(a, b):
    return mix64(mix64(a) ^ ((b + 0x632BE59BD9B4E019 + (a << 6) + (a >> 2)) & M64))


def dexp(x):
    LN2_HI, LN2_LO, INV_LN2 = 6.93147180369123816490e-01, 1.90821492927058770002e-10, 1.44269504088896338700e+00
    if x > 709.782712893384: return float("inf")
    if x < -745.1332191019412: return 0.0
    kd = eround(x * INV_LN2)
    return scale(exp_poly((x - kd * LN2_HI) - kd * LN2_LO), int(kd))


def dlog2(x):
    b = struct.unpack("<Q", struct.pack("<d", x))[0]
    e = b >> 52
    if e == 0:
        b = struct.unpack("<Q", struct.pack("<d", x * pow2i(64)))[0]
        e = (b >> 52) - 64
    e -= 1023
    m = struct.unpack("<d", struct.pack("<Q", (b & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000))[0]
    if m > 1.4142135623730951:
        m *= 0.5
        e += 1
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    t = 1.0 / 25.0
    for c in (23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3):
        t = t * s2 + 1.0 / c
    t = t * s2 + 1.0
    return float(e) + (2.0 * s * t) * 1.44269504088896338700e+00


def cascade_values(rows, cols, K, seed):
    """the initial VALUE plane: positions drawn as engram_cascade_init draws them"""
    base = mix64((seed * 0x9E3779B97F4A7C15 + 0xFFFFFFFF) & M64)
    out = np.empty(rows * cols, np.int64)
    top = 3 * K - 1
    for i in range(rows * cols):
        u = F32(mix64(i ^ base) >> 40) * F32(1.0 / 16777216.0) * F32(3 * K)
        p = min(int(u), top)
        out[i] = value(p, K)
    return out.reshape(rows, cols)


class Expert:
    def __init__(self, width, hidden, orders, blog2, K, seed, ctx_seed=0x435458):
        self.W, self.hidden, self.orders, self.blog2, self.ctx_seed = width, hidden, orders, blog2, ctx_seed
        F = orders << blog2
        self.bag = cascade_values(F, width, K, mix2(seed, 1))
        self.hid = [cascade_values(width, width, K, mix2(seed, 2 + i)) for i in range(hidden)]
        self.head = cascade_values(256, width, K, mix2(seed, 99))
        ng = (hidden + 1) * width + 256
        # exact binary fractions in (0.25, 1.75] and [-0.625, 0.625]
        self.gain = [F32(0.25 + ((i * 37) % 23 + 1) / 16.0) for i in range(ng)]
        self.bias = [F32(((v * 13) % 11 - 5) / 8.0) for v in range(256)]

    def ids(self, text, t):
        h, out = self.ctx_seed, []
        for k in range(1, self.orders + 1):
            b = text[t - k] if t >= k else 256
            h = mix2(h, b)
            out.append(((k - 1) << self.blog2) | (h & ((1 << self.blog2) - 1)))
        return out

    @staticmethod
    def norm(z, sc):
        ss = 0.0
        for v in z:
            d = float(int(v)) * sc
            ss += d * d
        r = math.sqrt(ss / len(z) + 1e-6)
        return [F32(float(int(v)) * sc / r) for v in z]

    def act(self, a, g0):
        mx = F32(0.0)
        ys = []
        for i, x in enumerate(a):
            y = x * self.gain[g0 + i]
            if not y > 0: y = F32(0.0)
            ys.append(y)
            if y > mx: mx = y
        s = float(mx) / 127.0 + 1e-12
        return np.array([int(eround(float(y) / s)) for y in ys], np.int64), s

    def logits(self, text, t):
        W = self.W
        alpha = 1.0 / math.sqrt(W)
        z = self.bag[self.ids(text, t)].sum(0)
        q, s = self.act(self.norm(z, 1.0 / math.sqrt(self.orders)), 0)
        for j in range(self.hidden):
            q, s = self.act(self.norm(self.hid[j] @ q, s * alpha), (j + 1) * W)
        lg = self.norm(self.head @ q, s * alpha)
        g0 = (self.hidden + 1) * W
        return [lg[v] * self.gain[g0 + v] + self.bias[v] for v in range(256)]

    def bits_at(self, text, t):
        lg = self.logits(text, t)
        m = float(lg[0])
        for x in lg[1:]:
            if float(x) > m: m = float(x)
        z = 0.0
        for x in lg: z += dexp(float(x) - m)
        return dlog2(z) - (float(lg[text[t]]) - m) * LOG2E


TEXT = ("Memory is the treasury and guardian of all things. — Cicero; "
        "été, naïve, 十二月. 12:34 -- ok!").encode("utf-8")

CASES = [  # width, hidden, orders, buckets_log2, K, seed
    (32, 0, 3, 6, 16, 11),
    (48, 1, 4, 5, 4, 12),
    (64, 2, 6, 7, 64, 13),
    (16, 0, 1, 4, 16, 14),
    (512, 0, 6, 12, 16, 0x534C4F57),       # the default: P2.1.1's choice
]


def main(out):
    L = ["/* GENERATED by tools/gen_expert_vectors.py -- DO NOT EDIT.",
         " * The expert's forward pass, computed by an independent Python implementation of engram_expert.h. */",
         "#ifndef ENGRAM_EXPERT_VECTORS_H", "#define ENGRAM_EXPERT_VECTORS_H",
         "static const uint8_t expert_text[] = { %s };" % ",".join(str(b) for b in TEXT),
         "#define EXPERT_TEXT_N %du" % len(TEXT),
         "typedef struct { unsigned width, hidden, orders, blog2, K; uint64_t seed;",
         "                 const double *bits; const unsigned *lpos; const float *logits; } expert_case;"]
    rows = []
    for ci, (W, hid, orders, blog2, K, seed) in enumerate(CASES):
        e = Expert(W, hid, orders, blog2, K, seed)
        bits = [e.bits_at(TEXT, t) for t in range(len(TEXT))]
        lpos = [0, len(TEXT) // 2, len(TEXT)]
        lg = []
        for t in lpos: lg += e.logits(TEXT, t)
        L.append("static const double ev%d_bits[] = { %s };" % (ci, ",".join(repr(b) for b in bits)))
        L.append("static const unsigned ev%d_lpos[] = { %s };" % (ci, ",".join("%du" % p for p in lpos)))
        L.append("static const float ev%d_logits[] = { %s };" % (ci, ",".join(repr(float(x)) + "f" if float(x) != 0.0 else "0.0f" for x in lg)))
        rows.append("    { %du, %du, %du, %du, %du, 0x%Xull, ev%d_bits, ev%d_lpos, ev%d_logits }," % (W, hid, orders, blog2, K, seed, ci, ci, ci))
        print("case %d: %d bits over %d bytes" % (ci, round(sum(bits)), len(TEXT)), file=sys.stderr)
    L.append("static const expert_case expert_cases[] = {")
    L += rows
    L += ["};", "#define EXPERT_CASES_N (sizeof expert_cases / sizeof expert_cases[0])", "#endif"]
    open(out, "w", newline="\n").write("\n".join(L) + "\n")
    print("wrote %s" % out)


if __name__ == "__main__":
    main(sys.argv[1])
