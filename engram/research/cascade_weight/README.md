# The Cascade weight

A weight that is, in one small integer word, **its value, its entire training state, and a record of how
consolidated it is** — trainable without a single float per weight, and shipped for inference at 1.6
bits per weight in a standard ONNX file that any runtime executes.

Research prototype for ENGRAM's slow store (Phase 2). Everything below was measured in this folder;
every number traces to a file in `results/`.

---

## First, what no weight format can do

A stored weight cannot hold more information than its bits. That is Shannon, and no format escapes it.
Every "more bits per weight" result in the literature is really one of three things: a format whose
bits are *better used* (codebooks matched to the weight distribution, lattice and trellis codes),
*fewer* weights doing the same work, or bits moved from one place to another. The frontier for
inference-only compression is crowded — trellis-coded quantization (QTIP, NeurIPS 2024) reports
near-optimal distortion at 2 bits per weight, and lattice codebooks up to the Leech lattice have been
applied to LLM weights (2026).

The place where bits are still wasted on an enormous scale is **training**. Every low-bit method in use
— BitNet b1.58 included — keeps a 16- or 32-bit *shadow* weight behind each ternary weight, plus
optimizer moments: 80–96 bits of state per weight to train a weight that ships in 1.6. And none of
that state knows anything about *which weights hold old knowledge*, which is why networks forget
catastrophically when they keep learning — the central problem for a system like ENGRAM that
consolidates into its own weights for its whole life.

The Cascade weight puts all of it in one word.

## The word

Three basins of K positions on one line:

```
position   0 ........ K-1 | K ........ 2K-1 | 2K ........ 3K-1
value           -1        |        0         |        +1
depth      K-1  ......  0 | lean -  ... + lean| 0  ......  K-1
```

stored as two separately addressable planes (the pattern/scale split of OCQ, applied to a training
state):

| plane | bits | read by |
|---|---|---|
| VALUE | 2 (ships packed at 1.6: five trits per byte) | inference — the only plane it reads |
| DEPTH | log2 K | training and consolidation |

| name | K | bits of state per weight |
|---|---|---|
| Cascade-4 | 4 | 4 |
| **Cascade-6** | **16** | **6** — the measured default |
| Cascade-8 | 64 | 8 (one byte) |

## The update — no float kept per weight

For each row of a weight matrix, the transient gradient `g` (a float, used and discarded) becomes a
desired move in positions, `u = -lr * g / rms(g_row)`. The move actually taken is `floor(|u|)` plus one
more with probability `frac(|u|)`: its **expected value is exactly `u`**, so small gradients accumulate
instead of vanishing — at most one basin per step. A move that brings a ±1 closer to 0 is first scaled
by `2^(-κ · depth · 4/K)`: **metaplasticity** — the consolidated-synapse principle of the cascade model
of Fusi, Drew & Abbott (Neuron 2005), in the form Laborieux et al. (Nature Communications 2021) applied
to binarised networks with float hidden weights — here in integer form. Moving deeper is never scaled.

The randomness is a counter-based hash of (seed, step, index) — splitmix64's finaliser, integer-only —
so a run is bit-reproducible: retraining seed 1 reproduced the measured model exactly (98.02% and
88.49%, the same bits).

## Results

MLP 784–512–512–10 (continual: 784–1024–1024–10), RMS-normalised layers, int8 activations for every
quantised arm, identical data order, schedule and seeds; arms differ **only** in how the weight
matrices are stored and updated. Learning rates and κ/m were chosen on validation data (the last 10k
training images; separate "dev" permutations for continual learning); every number below is on data
never used for any choice.

### 1. One task at a time — test accuracy, 3 seeds (`results/single_task_test.jsonl`)

| arm | training state per weight | inference bits | MNIST | Fashion-MNIST |
|---|---|---|---|---|
| FP32 + Adam | 96 | 32 | 98.50 ± 0.15 | 89.87 ± 0.05 |
| BitNet-style (float shadow + Adam) | 96 | 1.6 | 98.41 ± 0.10 | 89.47 ± 0.05 |
| Cascade-8 | **8** | 1.6 | 98.15 ± 0.03 | 88.93 ± 0.10 |
| **Cascade-6** | **6** | 1.6 | 98.04 ± 0.02 | 88.53 ± 0.05 |
| Cascade-4 | **4** | 1.6 | 96.76 ± 0.07 | 87.03 ± 0.29 |

On a single task the Cascade weight **trails** float-shadow training: Cascade-6 by 0.4 (MNIST) and
0.9 (Fashion) points, Cascade-8 by 0.3 and 0.5, for 12–16× less training state per weight. The 4-bit
word loses 1.6–2.4 points: 3.6 bits of accumulated history is too little (the depth plane saturates).

### 2. Learning without forgetting — permuted MNIST, 5 tasks in sequence, no replay (`results/continual_test.jsonl`)

Mean accuracy over all five tasks after the last one, 3 seeds, fresh permutations:

| arm | training state per weight | mean, all tasks | first task retained | last task learned |
|---|---|---|---|---|
| FP32 + Adam | 96 | 0.802 ± 0.023 | 0.404 | 0.978 |
| BitNet-style | 96 | 0.817 ± 0.011 | 0.428 | 0.976 |
| Laborieux metaplasticity (float shadow), m = 2 | 96 | 0.919 ± 0.006 | 0.775 | 0.963 |
| **Cascade-6, κ = 0.5** | **6** | **0.939 ± 0.001** | **0.963** | 0.958 |
| Cascade-6, κ = 0 — control | 6 | 0.471 ± 0.034 | 0.105 | 0.978 |

**Cascade-6 forgets least, on every seed** (+0.015, +0.021, +0.024 over the float-hidden metaplastic
arm), keeping the first task at 96% where that arm keeps 78%, at one sixteenth of the training state.
The control prices the mechanism: without its depth plane the same word collapses to 0.47. Bounded,
discrete synapses without metaplasticity forget fast — the problem the cascade model was proposed to
solve — and this is that problem, reproduced, and then removed by the depth plane.

### 3. Runs anywhere ONNX runs (`results/onnx_export_*.txt`)

`export_onnx.py` writes the trained network as a standard ONNX file (opset 18) whose weights are the
**packed trits themselves** — 668,672 weights in 133,735 bytes, exactly 1.600 bits per weight, a
142 KB file (FP32 would be 2.67 MB) — decoded inside the graph by stock operators only (Cast, Div, Mod,
Sub, Reshape, MatMul, ...). No custom kernel exists. onnxruntime 1.30 reproduces the numpy reference's
accuracy exactly (98.02%, 88.49%) with 100% prediction agreement.

## What is new here, and what is not

**Not new, and credited:** ternary weights (BitNet b1.58); stochastic rounding; counter-based random
numbers; metaplasticity for binarised networks (Laborieux et al. 2021) and the cascade synapse (Fusi et
al. 2005); integerised training with 4-bit latent weights (TPAMI 2025); Bop's point that binary weights
need no latent weight, only inertia (Helwegen et al., NeurIPS 2019).

**What this adds, as far as the searches in this session found:** those pieces had not been put in
one word — a single integer per weight that is simultaneously the inference weight (its own
separately addressable plane), the complete training state (no shadow weight, no moments), and a
metaplastic consolidation depth — trained with an unbiased, reproducible integer update, and shown to
**out-retain the float-hidden metaplastic method at 1/16 of the state**. That is a claim about this
benchmark at this scale, not yet a general one.

## Limits — named, with the attack beside each

- **Small models, one benchmark family.** MLPs on MNIST-scale data. Next: a character-level language
  model on ENGRAM's own corpora, and class-incremental splits, which are harder than permuted tasks.
- **Single-task accuracy trails float shadows by 0.3–0.9 points.** The measured lever is state bits
  (4 → 6 → 8 closes most of it); per-row adaptivity is the untested one.
- **Training is not bit-identical ACROSS platforms yet.** The state update and its random draws are
  exact and deterministic; the float gradients feeding them come from each platform's BLAS. The route:
  int8 activations × ternary weights stay below 2^24, so float32 GEMMs of them are exact integer
  arithmetic on every device — gradients can be made exact the same way. Not done yet.
- **Inference logits differ across runtimes by up to 0.018** (the float normalisation epilogue);
  predictions agree 100%.
- **Speed.** In numpy, continual-learning runs took ~1.7× as long end to end as the Adam arms (450 s vs
  270 s; the hash and the passes over the state). A C kernel is Phase 2 work.

## Reproduce

```
./fetch_data.sh                              # downloads MNIST + Fashion-MNIST, verifies SHA-256
python3 check.py                             # round-trips, RNG, gradient check (proven able to fail)
./runjobs.sh jobs_k.txt val 10 1             # validation sweeps (also jobs_val.txt)
./runfinal.sh jobs_final.txt                 # single-task test runs, 3 seeds, models saved
python3 cl.py cascade16:0.5 test 1 1         # continual learning, one arm and seed
python3 export_onnx.py runs/model_cascade16_mnist_1.npz cascade6.onnx mnist
```

Set `OPENBLAS_NUM_THREADS=1` when running several jobs at once: four processes each launching four BLAS
threads on four cores made an epoch take 100 s instead of 5.

Note on the record: the first Cascade-4 sweep capped every move at one position per step, which turned
large gradients into sign-SGD. It was found, fixed (the unbiased multi-position move above), and every
Cascade arm was re-run; the superseded runs are not in `results/`.
