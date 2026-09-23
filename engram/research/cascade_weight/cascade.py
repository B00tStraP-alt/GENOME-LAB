"""The Cascade weight: a small integer word that is, at once, a weight, its whole training state, and a
record of how consolidated it is.

THE WORD. K positions per basin, three basins in a line (3K positions):
    positions 0 .. K-1        value -1   (position 0 is the deepest -1)
    positions K .. 2K-1       value  0   (the lean: which side it lies closer to)
    positions 2K .. 3K-1      value +1   (position 3K-1 is the deepest +1)
stored as two separately addressable planes (after OCQ's pattern/scale split):
    VALUE plane   2 bits            {-1, 0, +1}; the ONLY plane inference reads (ships at 1.6 bits)
    DEPTH plane   log2(K) bits      distance from flipping (or the lean, for 0)
    K = 4   -> Cascade-4   (4 bits per weight)
    K = 16  -> Cascade-6   (6 bits per weight)   the measured default
    K = 64  -> Cascade-8   (8 bits per weight, one byte)

THE UPDATE. No float is kept per weight. The transient gradient g of a row is normalised by the row's
RMS; u = -lr * g / rms is the desired move in positions. The move taken is floor(|u|) plus one more
with probability frac(|u|) -- so its EXPECTED value is exactly u -- capped at one basin per step.
A move that brings a +-1 closer to 0 is first scaled by 2^(-kappa * depth * 4 / K): metaplasticity
(Fusi, Drew & Abbott 2005; Laborieux et al. 2021) in integer form. Moving deeper is never scaled.
The randomness is a counter-based hash of (seed, step, index) -- splitmix64's finaliser -- so a run is
bit-reproducible, and the draw sequence is identical on every platform.
"""
import gzip
import numpy as np

U64 = np.uint64
M64 = (1 << 64) - 1


# ---- data ------------------------------------------------------------------------------------------
def load_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    nd = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
    return np.frombuffer(raw, np.uint8, offset=4 + 4 * nd).reshape(dims)


def load(name, root):
    xtr = load_idx(f"{root}/{name}-train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    ytr = load_idx(f"{root}/{name}-train-labels-idx1-ubyte.gz").astype(np.int64)
    xte = load_idx(f"{root}/{name}-t10k-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255.0
    yte = load_idx(f"{root}/{name}-t10k-labels-idx1-ubyte.gz").astype(np.int64)
    return xtr, ytr, xte, yte


# ---- counter-based RNG: splitmix64's finaliser, integer-only, identical on every platform ---------
def mix64(x):
    x = x + U64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> U64(27))) * U64(0x94D049BB133111EB)
    return x ^ (x >> U64(31))


def uniform24(seed, step, n):
    """n uniforms in [0,1) with 24-bit resolution -- exact in float32 -- from (seed, step, index)."""
    with np.errstate(over="ignore"):
        base = mix64(np.array([(seed * 0x9E3779B97F4A7C15 + step) & M64], dtype=U64))[0]
        x = mix64(np.arange(n, dtype=U64) ^ base)
    return (x >> U64(40)).astype(np.float32) * np.float32(2.0 ** -24)


# ---- the word ---------------------------------------------------------------------------------------
NPOS = 12


def value_of(p):
    return (p >= 8).astype(np.int8) - (p <= 3).astype(np.int8)


def depth_of(p):
    """0..3: for -1 and +1 the distance from the flip boundary; for 0 the lean position."""
    return np.where(p <= 3, 3 - p, np.where(p >= 8, p - 8, p - 4)).astype(np.uint8)


def pack4(p):
    """Two planes in one byte per weight pair: value code (2 bits) | depth (2 bits) per nibble."""
    v = value_of(p)
    vcode = np.where(v == 0, 0, np.where(v > 0, 1, 3)).astype(np.uint8)   # 00 zero, 01 +1, 11 -1
    nib = (vcode | (depth_of(p) << 2)).astype(np.uint8).ravel()
    if nib.size % 2:
        nib = np.concatenate([nib, np.zeros(1, np.uint8)])
    return (nib[0::2] | (nib[1::2] << 4)).astype(np.uint8)


def unpack4(b, shape):
    nib = np.empty(b.size * 2, np.uint8)
    nib[0::2] = b & 0x0F
    nib[1::2] = b >> 4
    nib = nib[: int(np.prod(shape))].reshape(shape)
    vcode, d = nib & 3, nib >> 2
    return np.where(vcode == 1, 8 + d, np.where(vcode == 3, 3 - d, 4 + d)).astype(np.uint8)


def pack_trits(v):
    """Inference plane at 1.6 bits: five trits per byte (3^5 = 243 <= 256)."""
    t = (v.ravel().astype(np.int16) + 1).astype(np.uint8)
    pad = (-t.size) % 5
    t = np.concatenate([t, np.ones(pad, np.uint8)]).reshape(-1, 5)
    return (t[:, 0] + 3 * t[:, 1] + 9 * t[:, 2] + 27 * t[:, 3] + 81 * t[:, 4]).astype(np.uint8)


def unpack_trits(b, shape):
    b = b.astype(np.int16)
    t = np.stack([b % 3, (b // 3) % 3, (b // 9) % 3, (b // 27) % 3, (b // 81) % 3], 1).ravel()
    return (t[: int(np.prod(shape))] - 1).astype(np.int8).reshape(shape)


def value_k(p, K):
    return (p >= 2 * K).astype(np.int8) - (p < K).astype(np.int8)


def depth_k(p, K):
    return np.where(p < K, K - 1 - p, np.where(p >= 2 * K, p - 2 * K, p - K)).astype(np.int32)


class Cascade4:
    """A weight matrix of Cascade words, shape (out, in). K positions per basin: K = 4 is the 4-bit word
    (value 2 bits + depth 2 bits); K = 16 is 6 bits; K = 64 is 8 bits. The value plane is always 2 bits."""

    def __init__(self, n_out, n_in, seed, kappa=0.0, K=4):
        self.shape = (n_out, n_in)
        self.seed = seed
        self.kappa = kappa
        self.K = K
        self.npos = 3 * K
        self.step = 0
        u = uniform24(seed, 0xFFFFFFFF, n_out * n_in).reshape(self.shape)
        self.p = np.minimum((u * self.npos).astype(np.int32), self.npos - 1).astype(np.uint8)

    def weights(self):
        return value_k(self.p, self.K).astype(np.float32)

    def update(self, g, lr):
        """g: gradient w.r.t. the ternary weight (transient float). Row-normalised, then one
        stochastically-rounded integer move per weight."""
        rms = np.sqrt(np.mean(g * g, axis=1, keepdims=True)) + np.float32(1e-12)
        u = -lr * g / rms                                     # desired move, in positions
        mag = np.minimum(np.abs(u), np.float32(self.K))       # at most one basin per step
        up = u > 0
        p = self.p
        if self.kappa > 0.0:
            v = value_k(p, self.K)
            d = depth_k(p, self.K).astype(np.float32) * np.float32(4.0 / self.K)   # depth on the K=4 scale
            toward = ((v > 0) & ~up) | ((v < 0) & up)         # a move that brings a +-1 closer to 0
            mag = np.where(toward, mag * np.exp2(-self.kappa * d), mag)
        self.step += 1
        r = uniform24(self.seed, self.step, p.size).reshape(p.shape)
        whole = np.floor(mag)
        steps = (whole + (r < (mag - whole))).astype(np.int16)   # E[steps] == mag exactly
        delta = np.where(up, steps, -steps)
        self.p = np.clip(p.astype(np.int16) + delta, 0, self.npos - 1).astype(np.uint8)
        return int((steps > 0).sum())
