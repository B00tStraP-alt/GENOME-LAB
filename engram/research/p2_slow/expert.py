#!/usr/bin/env python3
"""
expert.py -- P2.1.1: what should a slow-store expert read, and how should its Cascade weights move?

A next-byte expert is trained on the TRAIN split of a corpus and judged on DEV, in bits per byte.
TEST is read only with --final, once, for the configurations the pre-registration names
(PREREG.md) -- it is never used to choose anything.

THE SPLIT. Paragraphs, interleaved: paragraph i goes to DEV if i % 10 == 8, TEST if i % 10 == 9,
TRAIN otherwise -- so every split holds every book (corpus_en.txt is four books in sequence; a split by
position would test on one unseen book and call it "held out").

THE FEATURES are the C library's (engram_ctx.h), computed the same way: for k = 1 .. orders,
h_k = mix2(h_{k-1}, byte[t-k] or 256 before the start), feature = (k-1) * buckets + h_k mod buckets,
with mix2 = engram_mix2 (splitmix64's finaliser, 64-bit wrap-around arithmetic).

THE BODY (every arm): input -> RMS norm + gain -> ReLU -> [hidden H x H]* -> head 256 x H -> RMS norm +
gain + bias -> softmax. Gains and bias: Adam. Matrices: Cascade-16 (kappa 0.5), int8 activations -- or,
in the float control, float32 + Adam everywhere (activations unquantised).

Arms (--input):
  hash    the ORDERS hashed contexts, a bag of rows of an H x F matrix
  window  the last CTX bytes as learned 16-d embeddings (float + Adam), concatenated

Usage:
  expert.py --input hash --width 256 --epochs 4 --lr 0.3 [--orders 6] [--buckets 1024] [--hidden 1]
            [--head cascade|float] [--norm unit|feature] [--sched cos|const] [--float] [--final]
            [--corpus en] [--seed 1]
  expert.py --counts [--corpus en]          # the count-model bars on DEV (and TEST with --final)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cascade_weight"))
from cascade import Cascade4, value_k, depth_k, mix64 as np_mix64, U64, M64  # noqa: E402

F32 = np.float32
DATA = os.path.join(HERE, "..", "..", "test", "data")


# ---- data ------------------------------------------------------------------------------------------
def split(corpus):
    raw = open(os.path.join(DATA, "corpus_%s.txt" % corpus), "rb").read()
    paras = [p for p in raw.split(b"\n") if p]
    parts = {"train": [], "dev": [], "test": []}
    for i, p in enumerate(paras):
        parts["dev" if i % 10 == 8 else "test" if i % 10 == 9 else "train"].append(p)
    return {k: np.frombuffer(b"\n".join(v) + b"\n", np.uint8) for k, v in parts.items()}


# ---- the C library's hashing, vectorised -------------------------------------------------------------
def mix2(a, b):
    with np.errstate(over="ignore"):
        return np_mix64(np_mix64(a) ^ (b + U64(0x632BE59BD9B4E019) + (a << U64(6)) + (a >> U64(2))))


def ctx_ids(data, pos, orders, blog2, seed=0x435458):
    out = np.zeros((len(pos), orders), np.int64)
    h = np.full(len(pos), seed, np.uint64)
    mask = U64((1 << blog2) - 1)
    for k in range(1, orders + 1):
        b = np.where(pos >= k, data[np.maximum(pos - k, 0)].astype(np.uint64), U64(256))
        h = mix2(h, b)
        out[:, k - 1] = ((h & mask).astype(np.int64)) + ((k - 1) << blog2)
    return out


# ---- the Cascade update restricted to touched columns (identical to the full update: W-proof in README)
def uniform24(seed, step, idx):
    with np.errstate(over="ignore"):
        base = np_mix64(np.array([(seed * 0x9E3779B97F4A7C15 + step) & M64], dtype=U64))[0]
        x = np_mix64(idx.astype(U64) ^ base)
    return (x >> U64(40)).astype(F32) * F32(2.0 ** -24)


def cascade_update_cols(W, cols, g, lr):
    """W: Cascade4 of shape (out, F); gradient zero outside `cols` (unique); g (out, len(cols)). Row RMS over
    all F columns; draws keyed r * F + c."""
    F = W.shape[1]
    rms = np.sqrt((g * g).sum(axis=1, keepdims=True) / F) + F32(1e-12)
    u = -lr * g / rms
    mag = np.minimum(np.abs(u), F32(W.K))
    up = u > 0
    p = W.p[:, cols]
    if W.kappa > 0.0:
        v = value_k(p, W.K)
        d = depth_k(p, W.K).astype(F32) * F32(4.0 / W.K)
        toward = ((v > 0) & ~up) | ((v < 0) & up)
        mag = np.where(toward, mag * np.exp2(-W.kappa * d), mag)
    W.step += 1
    idx = (np.arange(W.shape[0])[:, None].astype(np.int64) * F + cols[None, :].astype(np.int64)).ravel()
    r = uniform24(W.seed, W.step, idx).reshape(p.shape)
    whole = np.floor(mag)
    steps = (whole + (r < (mag - whole))).astype(np.int16)
    W.p[:, cols] = np.clip(p.astype(np.int16) + np.where(up, steps, -steps), 0, W.npos - 1).astype(np.uint8)


def cascade_update_rows(W, rows, g, lr):
    """W: Cascade4 of shape (F, out) -- rows = FEATURES; g (len(rows), out). Row RMS over each feature's row."""
    rms = np.sqrt(np.mean(g * g, axis=1, keepdims=True)) + F32(1e-12)
    u = -lr * g / rms
    mag = np.minimum(np.abs(u), F32(W.K))
    up = u > 0
    p = W.p[rows]
    if W.kappa > 0.0:
        v = value_k(p, W.K)
        d = depth_k(p, W.K).astype(F32) * F32(4.0 / W.K)
        toward = ((v > 0) & ~up) | ((v < 0) & up)
        mag = np.where(toward, mag * np.exp2(-W.kappa * d), mag)
    W.step += 1
    idx = (rows[:, None].astype(np.int64) * W.shape[1] + np.arange(W.shape[1])[None, :]).ravel()
    r = uniform24(W.seed, W.step, idx).reshape(p.shape)
    whole = np.floor(mag)
    steps = (whole + (r < (mag - whole))).astype(np.int16)
    W.p[rows] = np.clip(p.astype(np.int16) + np.where(up, steps, -steps), 0, W.npos - 1).astype(np.uint8)


# ---- layers ------------------------------------------------------------------------------------------
class Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape, F32); self.v = np.zeros(shape, F32); self.t = 0; self.lr = lr

    def step(self, w, g, sc):
        self.t += 1
        self.m = F32(0.9) * self.m + F32(0.1) * g
        self.v = F32(0.999) * self.v + F32(0.001) * g * g
        mh = self.m / F32(1 - 0.9 ** self.t); vh = self.v / F32(1 - 0.999 ** self.t)
        w -= F32(self.lr * sc) * mh / (np.sqrt(vh) + F32(1e-8))


def quant8(x):
    s = np.max(np.abs(x), axis=1, keepdims=True) / F32(127.0) + F32(1e-12)
    return np.rint(x / s).astype(F32), s


class Dense:
    """a weight matrix, Cascade (int8 activations) or float32 + Adam (the control)"""
    def __init__(self, n_out, n_in, seed, lr, flt):
        self.flt, self.lr = flt, lr
        if flt:
            rng = np.random.default_rng(seed)
            self.W = (rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)).astype(F32)
            self.opt = Adam(self.W.shape, 2e-3); self.alpha = F32(1.0)
        else:
            self.C = Cascade4(n_out, n_in, seed, 0.5, 16); self.alpha = F32(1 / np.sqrt(n_in))

    def fwd(self, x):
        if self.flt:
            self.x = x; return x @ self.W.T
        self.Wq = self.C.weights(); xq, s = quant8(x); self.x = xq * s
        return (self.x @ self.Wq.T) * self.alpha

    def bwd(self, dz, sc):
        dz = dz * self.alpha
        dW = dz.T @ self.x
        dx = dz @ (self.W if self.flt else self.Wq)
        if self.flt: self.opt.step(self.W, dW, sc)
        else: self.C.update(dW, F32(self.lr * sc))
        return dx


class Norm:
    def __init__(self, n):
        self.g = np.ones(n, F32); self.opt = Adam(n, 2e-3)

    def fwd(self, z):
        self.z = z; self.r = np.sqrt(np.mean(z * z, 1, keepdims=True) + F32(1e-6)); self.n = z / self.r
        return self.n * self.g

    def bwd(self, dy, sc):
        dg = (dy * self.n).sum(0); u = dy * self.g
        dz = u / self.r - self.z * np.sum(u * self.z, 1, keepdims=True) / (self.z.shape[1] * self.r ** 3)
        self.opt.step(self.g, dg, sc)
        return dz


class Expert:
    def __init__(self, a):
        self.a = a
        H, seed = a.width, a.seed
        if a.input == "hash":
            self.F = a.orders << a.buckets_log2
            if a.float:
                rng = np.random.default_rng(seed + 7)
                self.E = (rng.standard_normal((self.F, H)) * 0.5).astype(F32); self.Eopt = Adam(self.E.shape, 2e-3)
            elif a.norm == "unit":
                self.B = Cascade4(H, self.F, seed + 7, 0.5, 16)          # H x F: rows = hidden units
            else:
                self.B = Cascade4(self.F, H, seed + 7, 0.5, 16)          # F x H: rows = features
        else:
            self.emb = (np.random.default_rng(seed).standard_normal((257, 16)) * 0.5).astype(F32)
            self.Eopt = Adam(self.emb.shape, 2e-3)
            self.L0 = Dense(H, a.ctx * 16, seed + 1, a.lr, a.float)
        self.N0 = Norm(H)
        self.hid = [(Dense(H, H, seed + 2 + i, a.lr, a.float), Norm(H)) for i in range(a.hidden)]
        self.head = Dense(256, H, seed + 99, a.lr, a.float or a.head == "float")
        self.Nh = Norm(256)
        self.b = np.zeros(256, F32); self.bopt = Adam(256, 2e-3)

    def fwd(self, data, pos):
        a = self.a
        if a.input == "hash":
            self.f = ctx_ids(data, pos, a.orders, a.buckets_log2)
            if a.float: z = self.E[self.f].sum(1)
            elif a.norm == "unit": z = self.B.weights()[:, self.f].sum(2).T.astype(F32)
            else: z = self.B.weights()[self.f].sum(1).astype(F32)
            z = z * F32(1 / np.sqrt(a.orders))
        else:
            idx = np.stack([np.where(pos - k >= 0, data[np.maximum(pos - k, 0)], 256) for k in range(1, a.ctx + 1)], 1)
            self.idx = idx
            z = self.L0.fwd(self.emb[idx].reshape(len(pos), -1))
        h = np.maximum(self.N0.fwd(z), 0); self.acts = [h]
        for L, N in self.hid:
            h = np.maximum(N.fwd(L.fwd(h)), 0); self.acts.append(h)
        return self.Nh.fwd(self.head.fwd(h)) + self.b

    def bwd(self, d, sc):
        a = self.a
        self.bopt.step(self.b, d.sum(0), sc)
        dh = self.head.bwd(self.Nh.bwd(d, sc), sc)
        for i in range(len(self.hid) - 1, -1, -1):
            L, N = self.hid[i]
            dh = L.bwd(N.bwd(dh * (self.acts[i + 1] > 0), sc), sc)
        dz = self.N0.bwd(dh * (self.acts[0] > 0), sc)
        if a.input == "hash":
            s = dz * F32(1 / np.sqrt(a.orders))
            feats, inv = np.unique(self.f.ravel(), return_inverse=True)
            g = np.zeros((len(feats), dz.shape[1]), F32)
            np.add.at(g, inv, np.repeat(s, a.orders, axis=0))
            if a.float:
                # sparse Adam (as PyTorch's SparseAdam): only the rows this batch touched move, with the global
                # step for the bias correction -- dense Adam over all F x H floats every step would cost 9 minutes
                # an epoch and decay the moments of rows no batch has seen
                o = self.Eopt; o.t += 1
                o.m[feats] = F32(0.9) * o.m[feats] + F32(0.1) * g
                o.v[feats] = F32(0.999) * o.v[feats] + F32(0.001) * g * g
                mh = o.m[feats] / F32(1 - 0.9 ** o.t); vh = o.v[feats] / F32(1 - 0.999 ** o.t)
                self.E[feats] -= F32(o.lr * sc) * mh / (np.sqrt(vh) + F32(1e-8))
            elif a.norm == "unit":
                cascade_update_cols(self.B, feats, np.ascontiguousarray(g.T), F32(a.lr * sc))
            else:
                cascade_update_rows(self.B, feats, g, F32(a.lr * sc))
        else:
            dx = self.L0.bwd(dz, sc).reshape(len(dz), a.ctx, 16)
            gE = np.zeros_like(self.emb); np.add.at(gE, self.idx, dx); self.Eopt.step(self.emb, gE, sc)


def xent(lg, y):
    m = lg.max(1, keepdims=True); e = np.exp(lg - m); p = e / e.sum(1, keepdims=True)
    loss = -np.log2(p[np.arange(len(y)), y] + 1e-30).sum(); p[np.arange(len(y)), y] -= 1
    return loss, p / len(y)


def bpb(model, data, lo=0, hi=None, bs=4096):
    hi = len(data) if hi is None else hi
    tot = 0.0
    for s in range(lo, hi, bs):
        pos = np.arange(s, min(hi, s + bs)); l, _ = xent(model.fwd(data, pos), data[pos]); tot += l
    return tot / (hi - lo)


# ---- count-model bars ---------------------------------------------------------------------------------
def counts(train, evals):
    from collections import defaultdict
    c0 = np.bincount(train, minlength=256) + 1; p0 = c0 / c0.sum()
    out = {"uniform": {k: 8.0 for k in evals}, "unigram": {k: float(-np.log2(p0[v]).mean()) for k, v in evals.items()}}
    for order in (1, 2, 3, 4):
        cnt = defaultdict(lambda: np.zeros(256))
        tb = bytes(train)
        for i in range(order, len(tb)): cnt[tb[i - order:i]][tb[i]] += 1
        res = {}
        for name, v in evals.items():
            vb = bytes(v); tot = 0.0
            for i in range(len(vb)):
                c = cnt.get(vb[max(0, i - order):i]) if i >= order else None
                N = c.sum() if c is not None else 0.0; d = (c > 0).sum() if c is not None else 0
                lam = N / (N + d) if N else 0.0
                p = lam * (c[vb[i]] / N if N else 0.0) + (1 - lam) * p0[vb[i]]
                tot += -math.log2(p)
            res[name] = tot / len(vb)
        out["order-%d counts (Witten-Bell to unigram)" % order] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="hash", choices=["hash", "window"])
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1)
    ap.add_argument("--orders", type=int, default=6)
    ap.add_argument("--buckets", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=12)
    ap.add_argument("--head", default="cascade", choices=["cascade", "float"])
    ap.add_argument("--norm", default="unit", choices=["unit", "feature"])
    ap.add_argument("--sched", default="cos", choices=["cos", "const"])
    ap.add_argument("--float", action="store_true")
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--corpus", default="en")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    a.buckets_log2 = int(math.log2(a.buckets))
    assert 1 << a.buckets_log2 == a.buckets
    S = split(a.corpus)
    if a.counts:
        ev = {"dev": S["dev"]}
        if a.final: ev["test"] = S["test"]
        print(json.dumps({"corpus": a.corpus, "counts": counts(S["train"], ev)}), flush=True)
        return
    model = Expert(a)
    tr = S["train"]; rng = np.random.default_rng(a.seed); bs = 128
    T = a.epochs * ((len(tr) - 1) // bs); t = 0; hist = []
    for ep in range(a.epochs):
        t0 = time.time()
        order = rng.permutation(np.arange(1, len(tr)))
        for b in range(len(order) // bs):
            pos = order[b * bs:(b + 1) * bs]
            sc = F32(0.5 * (1 + np.cos(np.pi * t / T))) if a.sched == "cos" else F32(1.0)
            _, d = xent(model.fwd(tr, pos), tr[pos]); model.bwd(d, sc); t += 1
        rec = {"epoch": ep + 1, "train_sample": float(bpb(model, tr, 0, 20000)), "dev": float(bpb(model, S["dev"])),
               "sec": time.time() - t0}
        hist.append(rec)
        print("  %s ep %d: train %.3f dev %.3f (%.0fs)" % (a.tag, ep + 1, rec["train_sample"], rec["dev"], rec["sec"]), file=sys.stderr, flush=True)
    out = {"tag": a.tag, "args": {k: v for k, v in vars(a).items() if k not in ("final", "counts")}, "history": hist,
           "best_dev": min(h["dev"] for h in hist), "final_dev": hist[-1]["dev"]}
    if a.final:
        out["test"] = float(bpb(model, S["test"]))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
