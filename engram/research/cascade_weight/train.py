"""Experiment harness. Arms differ ONLY in how the three weight matrices are stored and updated:
    fp32     float32 weights, Adam                          (ceiling)
    bitnet   float32 shadow weights, absmean ternary forward, straight-through, Adam  (state of the art)
    cascade  Cascade-4 words, no float per weight
Everything else -- architecture, int8 activations for the quantised arms, per-feature gains, bias,
schedule, data order, seeds -- is identical."""
import sys, time, json
import numpy as np
from cascade import Cascade4, load, value_of

F32 = np.float32


class Adam:
    def __init__(self, shape, lr):
        self.m = np.zeros(shape, F32); self.v = np.zeros(shape, F32); self.t = 0; self.lr = lr

    def step(self, w, g, lr_scale):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
        w -= (self.lr * lr_scale) * mh / (np.sqrt(vh) + 1e-8)


def quant8(x):
    s = np.max(np.abs(x), axis=1, keepdims=True) / F32(127.0) + F32(1e-12)
    return np.rint(x / s).astype(F32), s


class Layer:
    def __init__(self, arm, n_out, n_in, seed, lr, kappa=0.0):
        self.arm, self.n_in = arm, n_in
        rng = np.random.default_rng(seed)
        self.alpha = F32(1.0 / np.sqrt(n_in))
        if arm.startswith("cascade"):
            K = int(arm[7:]) if len(arm) > 7 else 4          # "cascade" = K 4, "cascade16", "cascade64"
            self.W = Cascade4(n_out, n_in, seed, kappa, K)
        else:
            self.W = (rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)).astype(F32)
            self.opt = Adam(self.W.shape, lr)
        self.lr = lr

    def wq(self):
        if self.arm.startswith("cascade"):
            return self.W.weights()
        if self.arm == "bitnet":
            beta = np.mean(np.abs(self.W)) + F32(1e-8)
            return np.clip(np.rint(self.W / beta), -1, 1).astype(F32)
        return self.W

    def forward(self, x):
        self.Wq = self.wq()
        if self.arm == "fp32":
            self.xh = x
        else:
            xq, s = quant8(x)
            self.xh = xq * s
        return (self.xh @ self.Wq.T) * self.alpha

    def backward(self, dz, lr_scale):
        dz = dz * self.alpha
        dW = dz.T @ self.xh
        dx = dz @ self.Wq
        if self.arm.startswith("cascade"):
            self.moves = self.W.update(dW, F32(self.lr * lr_scale))
        else:
            self.opt.step(self.W, dW, lr_scale)
        return dx

    def bits(self):
        if self.arm.startswith("cascade"):
            return self.W.p.size, 2 + int(np.log2(self.W.K)), 1.6
        n = self.W.size
        train = {"fp32": 96, "bitnet": 96}[self.arm]
        infer = {"fp32": 32, "bitnet": 1.6}[self.arm]
        return n, train, infer


class Net:
    def __init__(self, arm, sizes, seed, lr, kappa=0.0, float_lr=2e-3):
        self.layers = [Layer(arm, sizes[i + 1], sizes[i], seed * 100 + i, lr, kappa) for i in range(len(sizes) - 1)]
        self.gain = [np.ones(n, F32) for n in sizes[1:]]
        self.gopt = [Adam(g.shape, float_lr) for g in self.gain]
        self.bias = np.zeros(sizes[-1], F32)
        self.bopt = Adam(self.bias.shape, float_lr)

    def forward(self, x):
        self.cache = []
        h = x
        L = len(self.layers)
        for i, (lay, g) in enumerate(zip(self.layers, self.gain)):
            z = lay.forward(h)
            r = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + F32(1e-6))
            n = z / r
            y = n * g
            self.cache.append((z, r, n))
            if i < L - 1:
                h = np.maximum(y, 0)
                self.cache[-1] = (z, r, n, y)
            else:
                h = y + self.bias
        return h

    def backward(self, dlogits, lr_scale):
        L = len(self.layers)
        dh = dlogits
        self.bopt.step(self.bias, dh.sum(0), lr_scale)
        for i in reversed(range(L)):
            c = self.cache[i]
            z, r, n = c[0], c[1], c[2]
            if i < L - 1:
                dh = dh * (c[3] > 0)
            dg = (dh * n).sum(0)
            u = dh * self.gain[i]
            dz = u / r - z * np.sum(u * z, axis=1, keepdims=True) / (z.shape[1] * r ** 3)
            self.gopt[i].step(self.gain[i], dg, lr_scale)
            dh = self.layers[i].backward(dz, lr_scale)


def xent(logits, y):
    m = logits.max(1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    p[np.arange(len(y)), y] -= 1
    return loss, p / len(y)


def accuracy(net, x, y, bs=2000):
    c = 0
    for i in range(0, len(x), bs):
        c += int((net.forward(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum())
    return c / len(x)


def train(net, xtr, ytr, epochs, seed, bs=128, xte=None, yte=None, log=True, perm=None):
    rng = np.random.default_rng(seed)
    T = epochs * (len(xtr) // bs)
    t = 0
    hist = []
    for ep in range(epochs):
        order = rng.permutation(len(xtr))
        t0 = time.time()
        for b in range(len(xtr) // bs):
            idx = order[b * bs:(b + 1) * bs]
            xb = xtr[idx] if perm is None else xtr[idx][:, perm]
            lr_scale = F32(0.5 * (1 + np.cos(np.pi * t / T)))
            logits = net.forward(xb)
            loss, d = xent(logits, ytr[idx])
            net.backward(d, lr_scale)
            t += 1
        if xte is not None:
            xt = xte if perm is None else xte[:, perm]
            acc = accuracy(net, xt, yte)
            hist.append(acc)
            if log:
                print(f"    epoch {ep + 1}: loss {loss:.4f} test acc {acc:.4f} ({time.time() - t0:.1f}s)", flush=True)
    return hist


if __name__ == "__main__":
    arm, dataset, lr = sys.argv[1], sys.argv[2], float(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    split = sys.argv[6] if len(sys.argv) > 6 else "test"
    xtr, ytr, xte, yte = load(dataset, "data")
    if split == "val":          # model selection never sees the test set
        xtr, ytr, xte, yte = xtr[:50000], ytr[:50000], xtr[50000:], ytr[50000:]
    net = Net(arm, [784, 512, 512, 10], seed, lr)
    print(f"{arm} {dataset} lr={lr} epochs={epochs} seed={seed}", flush=True)
    hist = train(net, xtr, ytr, epochs, seed, xte=xte, yte=yte)
    import os
    if os.environ.get("SAVE"):
        np.savez(os.environ["SAVE"], arm=arm, sizes=np.array([784, 512, 512, 10]),
                 gains=np.array(net.gain, dtype=object), bias=net.bias,
                 alphas=np.array([l.alpha for l in net.layers], F32),
                 **{f"p{i}": (l.W.p if arm.startswith("cascade") else l.wq()) for i, l in enumerate(net.layers)})
        # (export expects the K = 4 word; other K export through their value plane the same way)
    nb = [l.bits() for l in net.layers]
    n = sum(a for a, _, _ in nb)
    print(json.dumps({"arm": arm, "dataset": dataset, "split": split, "lr": lr, "seed": seed, "final": hist[-1], "best": max(hist),
                      "weights": n, "train_bits_per_weight": nb[0][1], "infer_bits_per_weight": nb[0][2]}))
