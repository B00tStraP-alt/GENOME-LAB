"""Continual learning: permuted MNIST, tasks in sequence, no replay of earlier data.
Arms:
    fp32              float weights, Adam
    bitnet            float shadow weights, ternary forward, Adam
    bitnet_meta:m     Laborieux et al. 2021 metaplasticity on the float shadow weight: an Adam step that
                      would SHRINK |W| is multiplied by 1 - tanh^2(m |W| / mean|W|)
    cascade:k         Cascade-4, kappa = k (0 = no metaplasticity)
Selection (m, k) on DEV permutations; reporting on TEST permutations never used for selection.
Metric: mean test accuracy over all tasks after the last task, and per task."""
import sys, json, time
import numpy as np
from cascade import load
from train import Net, train, accuracy, Adam, F32

N_TASKS = 5
EPOCHS = 2
SIZES = [784, 1024, 1024, 10]


def perms(kind, n):
    base = 100 if kind == "dev" else 200
    return [np.arange(784) if i == 0 and kind == "test" else np.random.default_rng(base + i).permutation(784)
            for i in range(n)]


def make(arm, seed, lr):
    kappa = 0.0
    name = arm
    if arm.startswith("cascade"):                       # "cascade:k", "cascade16:k", ...
        name, k = (arm.split(":") + ["0"])[:2]
        kappa = float(k)
    if arm.startswith("bitnet_meta:"):
        name = "bitnet"
    net = Net(name, SIZES, seed, lr, kappa=kappa)
    if arm.startswith("bitnet_meta:"):
        m = float(arm.split(":")[1])
        for lay in net.layers:
            lay.opt = MetaAdam(lay.W.shape, lr, m)
    return net


class MetaAdam(Adam):
    def __init__(self, shape, lr, m):
        super().__init__(shape, lr); self.meta = m

    def step(self, w, g, lr_scale):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        mh = self.m / (1 - 0.9 ** self.t); vh = self.v / (1 - 0.999 ** self.t)
        d = -(self.lr * lr_scale) * mh / (np.sqrt(vh) + 1e-8)
        beta = np.mean(np.abs(w)) + F32(1e-8)
        shrink = (d * w) < 0
        f = F32(1.0) - np.tanh(self.meta * np.abs(w) / beta) ** 2
        w += np.where(shrink, d * f, d)


if __name__ == "__main__":
    arm, kind, lr, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    xtr, ytr, xte, yte = load("mnist", "data")
    P = perms(kind, N_TASKS)
    net = make(arm, seed, lr)
    t0 = time.time()
    after = []                        # after[t][j]: accuracy on task j after training task t
    for t in range(N_TASKS):
        train(net, xtr, ytr, EPOCHS, seed * 1000 + t, perm=P[t], log=False)
        after.append([accuracy(net, xte[:, P[j]], yte) for j in range(t + 1)])
        print(f"  {arm} after task {t + 1}: " + " ".join(f"{a:.4f}" for a in after[-1]), flush=True)
    final = after[-1]
    print(json.dumps({"arm": arm, "kind": kind, "lr": lr, "seed": seed, "final": final,
                      "mean": float(np.mean(final)), "first_task": final[0], "secs": time.time() - t0}))
