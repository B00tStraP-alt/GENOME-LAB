#!/usr/bin/env python3
"""
ref_expert.py -- the independent Python implementation of the slow store's expert that the C is held to:
the forward pass (engram_expert.h), and the backward pass as engram_expert.h's "THE BACKWARD PASS,
EXACTLY" specifies it -- every float32 operation in numpy float32, every double one in Python floats,
every sum in the order the specification states (sums are not associative in floating point, so the
order IS the specification).

It also carries the evidence that those backward FORMULAS are right: `shadow_gradcheck` builds the same
network in float64 with the straight-through quantisation made the identity it pretends to be, and
compares the formulas' gradient with central finite differences of the loss, for every kind of
parameter. tools/gen_grad_vectors.py refuses to write vectors if it fails.
"""
import math
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_cascade_vectors import mix64, exp_poly, scale, eround, pow2i  # noqa: E402

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


def cascade_positions_scalar(rows, cols, K, seed):
    """engram_cascade_init's draw, one weight at a time, as engram_cascade.h states it"""
    base = mix64((seed * 0x9E3779B97F4A7C15 + 0xFFFFFFFF) & M64)
    top = 3 * K - 1
    out = np.empty(rows * cols, np.int64)
    for i in range(rows * cols):
        u = F32(mix64(i ^ base) >> 40) * F32(1.0 / 16777216.0) * F32(3 * K)
        out[i] = min(int(u), top)
    return out.reshape(rows, cols)


def _mix64_np(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def cascade_positions(rows, cols, K, seed):
    """the same draw in numpy (uint64 arithmetic wraps as C's does; float32 products in the same order),
    for the default expert's 12.7 M weights; proven equal to the scalar draw by the generators"""
    base = np.uint64(mix64((seed * 0x9E3779B97F4A7C15 + 0xFFFFFFFF) & M64))
    i = np.arange(rows * cols, dtype=np.uint64)
    u = (_mix64_np(i ^ base) >> np.uint64(40)).astype(F32) * F32(1.0 / 16777216.0) * F32(3 * K)
    return np.minimum(u.astype(np.int64), 3 * K - 1).reshape(rows, cols)


def values_of(pos, K):
    return (pos >= 2 * K).astype(np.int64) - (pos < K).astype(np.int64)


class Expert:
    def __init__(self, width, hidden, orders, blog2, K, seed, kappa=0.5, ctx_seed=0x435458):
        self.W, self.hidden, self.orders, self.blog2, self.K, self.kappa = width, hidden, orders, blog2, K, kappa
        self.ctx_seed = ctx_seed
        F = orders << blog2
        self.F = F
        self.seeds = {"bag": mix2(seed, 1), "hid": [mix2(seed, 2 + i) for i in range(hidden)], "head": mix2(seed, 99)}
        self.bag_p = cascade_positions(F, width, K, self.seeds["bag"])
        self.hid_p = [cascade_positions(width, width, K, s) for s in self.seeds["hid"]]
        self.head_p = cascade_positions(256, width, K, self.seeds["head"])
        self.bag_v = values_of(self.bag_p, K)
        self.hid_v = [values_of(p, K) for p in self.hid_p]
        self.head_v = values_of(self.head_p, K)
        ng = (hidden + 1) * width + 256
        self.gain = [F32(1.0)] * ng
        self.bias = [F32(0.0)] * 256

    def set_pattern(self, signed=False):
        """exact binary fractions: gains in (0.25, 1.75], every fifth one negated if signed (a trained gain
        can cross zero -- then ReLU's mask and the norm's output differ in sign), biases in [-0.625, 0.625]"""
        self.gain = [F32(0.25 + ((i * 37) % 23 + 1) / 16.0) for i in range(len(self.gain))]
        if signed: self.gain = [-x if i % 5 == 3 else x for i, x in enumerate(self.gain)]
        self.bias = [F32(((v * 13) % 11 - 5) / 8.0) for v in range(256)]

    # ---- forward, with the caches the backward needs --------------------------------------------------
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
        return [F32(float(int(v)) * sc / r) for v in z], r

    def act(self, nrm, g0):
        mx = F32(0.0)
        a = []
        for i, x in enumerate(nrm):
            y = x * self.gain[g0 + i]
            if not y > 0: y = F32(0.0)
            a.append(y)
            if y > mx: mx = y
        s = float(mx) / 127.0 + 1e-12
        return a, np.array([int(eround(float(y) / s)) for y in a], np.int64), s

    def forward(self, text, t):
        W = self.W
        alpha = 1.0 / math.sqrt(W)
        c = {"ids": self.ids(text, t), "layers": []}
        z = self.bag_v[c["ids"]].sum(0)
        sc = 1.0 / math.sqrt(self.orders)
        for j in range(self.hidden + 1):
            nrm, r = self.norm(z, sc)
            a, q, s = self.act(nrm, j * W)
            c["layers"].append({"z": z, "sc": sc, "r": r, "n": nrm, "a": a, "q": q, "s": s})
            if j < self.hidden:
                z = self.hid_v[j] @ q
                sc = s * alpha
        zh = self.head_v @ c["layers"][-1]["q"]
        sch = c["layers"][-1]["s"] * alpha
        nh, rh = self.norm(zh, sch)
        g0 = (self.hidden + 1) * W
        lg = [nh[v] * self.gain[g0 + v] + self.bias[v] for v in range(256)]
        c["head"] = {"z": zh, "sc": sch, "r": rh, "n": nh}
        c["logits"] = lg
        return c

    @staticmethod
    def softmax_bits(lg, y):
        m = float(lg[0])
        for x in lg[1:]:
            if float(x) > m: m = float(x)
        e = [dexp(float(x) - m) for x in lg]
        Z = 0.0
        for x in e: Z += x
        return [x / Z for x in e], dlog2(Z) - (float(lg[y]) - m) * LOG2E

    # ---- the backward pass (engram_expert.h: THE BACKWARD PASS, EXACTLY) ------------------------------
    @staticmethod
    def norm_back(dn, gain, g0, z, sc, r, nrm, dg):
        """dn: dL/dy (float32), y = n * gain, n = norm(z, sc). Accumulates dL/dgain into dg[g0 + i] and
        returns dL/dv (float32), v = z * sc."""
        n = len(z)
        r3 = r * r * r
        suv = 0.0
        for i in range(n):
            u = float(dn[i]) * float(gain[g0 + i])
            suv += u * (float(int(z[i])) * sc)
        dv = []
        for i in range(n):
            dg[g0 + i] = dg[g0 + i] + dn[i] * nrm[i]
            u = float(dn[i]) * float(gain[g0 + i])
            dv.append(F32(u / r - (float(int(z[i])) * sc) * suv / (n * r3)))
        return dv

    def backward(self, text, positions):
        """Forward and backward over a batch; returns the loss in bits and the gradients, accumulated over
        the positions in the order given."""
        W, B = self.W, len(positions)
        alpha = 1.0 / math.sqrt(W)
        g0h = (self.hidden + 1) * W
        g = {"bias": [F32(0.0)] * 256, "gain": [F32(0.0)] * len(self.gain),
             "head": np.zeros((256, W), F32), "hid": [np.zeros((W, W), F32) for _ in range(self.hidden)],
             "bag": {}}
        head_v, hid_v = self.head_v, self.hid_v
        bits = 0.0
        for t in positions:
            c = self.forward(text, t)
            y = text[t]
            p, b = self.softmax_bits(c["logits"], y)
            bits += b
            dl = [F32((p[v] - (1.0 if v == y else 0.0)) / B) for v in range(256)]
            for v in range(256):
                g["bias"][v] = g["bias"][v] + dl[v]
            H = c["head"]
            dv = self.norm_back(dl, self.gain, g0h, H["z"], H["sc"], H["r"], H["n"], g["gain"])
            last = c["layers"][-1]
            xs = np.array([F32(float(int(q)) * last["s"]) for q in last["q"]], F32)
            dva = np.array([F32(float(d) * alpha) for d in dv], F32)
            g["head"] += dva[:, None] * xs[None, :]
            dx = matvec_t(head_v, dva)
            for j in range(self.hidden, -1, -1):
                L = c["layers"][j]
                da = [dx[i] if L["a"][i] > 0 else F32(0.0) for i in range(W)]
                dv = self.norm_back(da, self.gain, j * W, L["z"], L["sc"], L["r"], L["n"], g["gain"])
                if j > 0:
                    prev = c["layers"][j - 1]
                    xs = np.array([F32(float(int(q)) * prev["s"]) for q in prev["q"]], F32)
                    dva = np.array([F32(float(d) * alpha) for d in dv], F32)
                    g["hid"][j - 1] += dva[:, None] * xs[None, :]
                    dx = matvec_t(hid_v[j - 1], dva)
                else:
                    c0 = 1.0 / math.sqrt(self.orders)
                    d0 = np.array([F32(float(d) * c0) for d in dv], F32)
                    for f in c["ids"]:
                        if f not in g["bag"]: g["bag"][f] = np.zeros(W, F32)    # from +0, as specified
                        g["bag"][f] += d0
        return bits, g


def matvec_t(values, d):
    """engram_cascade_matvec_t: y[c] = sum over r ascending of +d[r] / -d[r] / nothing, zero d skipped"""
    rows, cols = values.shape
    y = np.zeros(cols, F32)
    for r in range(rows):
        if d[r] == 0: continue
        pos = values[r] > 0
        neg = values[r] < 0
        y[pos] = y[pos] + d[r]
        y[neg] = y[neg] - d[r]
    return y


# ---- the evidence that the backward formulas are right -----------------------------------------------
def shadow_gradcheck(width=24, hidden=1, orders=3, blog2=4, K=16, seed=5, n_pos=6, eps=1e-6, plant=False):
    """The same network in float64, straight-through quantisation made the identity, weights as real
    numbers initialised to the ternary values: the backward formulas above, re-derived in float64, against
    central finite differences of the loss (nats) for every parameter kind. Returns the worst relative
    error per kind."""
    e = Expert(width, hidden, orders, blog2, K, seed)
    e.set_pattern(signed=True)
    rng = np.random.default_rng(1)
    text = bytes(rng.integers(32, 127, 40).tolist())
    positions = list(range(3, 3 + n_pos))
    W = width
    # Real-valued weights: the ternary values plus a little noise. Sums of ternary values are often EXACTLY
    # zero, and a ReLU at exactly zero has no derivative: a central difference there measures half a slope,
    # which is the kink, not the formula (found by this check's first run). Off the lattice, no pre-activation
    # sits on a kink, and the formulas -- which do not depend on the weights' values -- are what is tested.
    jit = np.random.default_rng(3)
    def real(p): return values_of(p, K).astype(np.float64) + jit.uniform(-0.05, 0.05, p.shape)
    P = {"bag": real(e.bag_p), "hid": [real(p) for p in e.hid_p], "head": real(e.head_p),
         "gain": np.array(e.gain, np.float64), "bias": np.array(e.bias, np.float64)}
    alpha = 1.0 / math.sqrt(W)
    c0 = 1.0 / math.sqrt(orders)

    def loss_and_grad(P, want_grad=True):
        L = 0.0
        G = {k: (np.zeros_like(v) if not isinstance(v, list) else [np.zeros_like(x) for x in v]) for k, v in P.items()}
        for t in positions:
            ids = e.ids(text, t)
            v = P["bag"][ids].sum(0) * c0
            cache = []
            x = None
            for j in range(hidden + 1):
                if j > 0: v = alpha * (P["hid"][j - 1] @ x)
                r = math.sqrt(np.mean(v * v) + 1e-6); n = v / r
                yv = n * P["gain"][j * W:(j + 1) * W]; a = np.maximum(yv, 0)
                cache.append((v, r, n, a, x)); x = a
            vh = alpha * (P["head"] @ x); rh = math.sqrt(np.mean(vh * vh) + 1e-6); nh = vh / rh
            g0 = (hidden + 1) * W
            lg = nh * P["gain"][g0:] + P["bias"]
            m = lg.max(); pz = np.exp(lg - m); Z = pz.sum(); p = pz / Z
            L += -(lg[text[t]] - m - math.log(Z))
            if not want_grad: continue
            dl = p.copy(); dl[text[t]] -= 1.0
            G["bias"] += dl
            G["gain"][g0:] += dl * nh
            du = dl * P["gain"][g0:]
            dvh = du / rh - vh * np.dot(du, vh) / (len(vh) * rh ** 3)
            G["head"] += np.outer(dvh * alpha, x)
            dx = (dvh * alpha) @ P["head"]
            for j in range(hidden, -1, -1):
                v, r, n, a, xin = cache[j]
                da = dx * (a > 0)
                G["gain"][j * W:(j + 1) * W] += da * n
                du = da * P["gain"][j * W:(j + 1) * W]
                dv = du / r - (0.0 if plant else 1.0) * v * np.dot(du, v) / (len(v) * r ** 3)
                if j > 0:
                    G["hid"][j - 1] += np.outer(dv * alpha, xin)
                    dx = (dv * alpha) @ P["hid"][j - 1]
                else:
                    for f in ids: G["bag"][f] += dv * c0
        return L, G

    L0, G = loss_and_grad(P)
    worst = {}

    def check(kind, arr, garr, idxs):
        w = 0.0
        for idx in idxs:
            old = arr[idx]
            arr[idx] = old + eps; lp, _ = loss_and_grad(P, False)
            arr[idx] = old - eps; lm, _ = loss_and_grad(P, False)
            arr[idx] = old
            fd = (lp - lm) / (2 * eps)
            an = garr[idx]
            # pass iff |fd - an| <= 1e-5 (|fd| + |an|) + 2e-8: relative where the gradient is large; an absolute
            # floor above the difference quotient's own rounding (a ~30-nat loss in doubles over a 2e-6 step:
            # a few 1e-9) where a unit the ReLU switched off reaches the loss only through its norm
            w = max(w, abs(fd - an) / (1e-5 * (abs(fd) + abs(an)) + 2e-8))
        worst[kind] = w

    rs = np.random.default_rng(2)
    check("bias", P["bias"], G["bias"], [(i,) for i in rs.integers(0, 256, 24)])
    check("head gain", P["gain"], G["gain"], [((hidden + 1) * W + i,) for i in rs.integers(0, 256, 24)])
    check("layer gains", P["gain"], G["gain"], [(i,) for i in rs.integers(0, (hidden + 1) * W, 24)])
    check("head weights", P["head"], G["head"], [tuple(x) for x in zip(rs.integers(0, 256, 24), rs.integers(0, W, 24))])
    if hidden:
        check("hidden weights", P["hid"][0], G["hid"][0], [tuple(x) for x in zip(rs.integers(0, W, 24), rs.integers(0, W, 24))])
    used = sorted({f for t in positions for f in e.ids(text, t)})
    check("bag rows", P["bag"], G["bag"], [(used[int(i)], int(j)) for i, j in zip(rs.integers(0, len(used), 24), rs.integers(0, W, 24))])
    return worst


if __name__ == "__main__":
    # R6: a planted error in one formula -- the norm's correction term dropped -- must be caught
    planted = shadow_gradcheck(hidden=1, plant=True)
    print("planted error: " + ", ".join("%s %.3f" % kv for kv in planted.items()),
          "CAUGHT" if max(planted.values()) > 1.0 else "MISSED -- THE CHECK CANNOT FAIL")
    for hid in (0, 1, 2):
        w = shadow_gradcheck(hidden=hid)
        print("hidden %d: " % hid + ", ".join("%s %.3f" % kv for kv in w.items()), "PASS" if max(w.values()) <= 1.0 else "FAIL")
