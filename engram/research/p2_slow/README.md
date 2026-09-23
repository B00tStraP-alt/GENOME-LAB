# P2.1.1 — what a slow-store expert reads, and how its weights move

Every number below is in `results/` (one JSON line per run in `runs.jsonl`, the count bars in
`counts_*.json`). The questions, their order and the decision rules were committed in `PREREG.md`
before the first run.

**Protocol.** `corpus_en.txt`, paragraphs interleaved into TRAIN (8 of every 10), DEV and TEST, so each
split holds all four books. Next-byte experts trained on TRAIN, batch 128; bits per byte. Choices on
DEV (best of 4 epochs, seed 1); TEST read once, after the last choice. `expert.py` computes the context
features exactly as `engram_ctx.c` does (checked: identical ids on a mixed ASCII/UTF-8 text).

## The chosen expert

| | |
|---|---|
| input | 6 hashed context orders x 4096 buckets, a bag of rows of a Cascade matrix |
| bag update | normalised per hidden unit over all features (zeros counted) |
| body | RMS norm + gain, ReLU, int8 -- no hidden layer |
| head | 256 x 512 Cascade, RMS norm + gain + bias |
| width | 512 |
| weights | Cascade-16 (6 bits of state a weight), kappa 0.5 |
| learning | constant rate 0.1 (a memory never "finishes" training), 2 epochs (best DEV epoch) |
| state | 6 x 4096 x 512 + 256 x 512 = 12.7 M weights, one byte each; 1,024 floats of gains and bias |

## TEST, read once

| | English TEST bits/byte |
|---|---|
| uniform | 8.000 |
| unigram | 4.435 |
| order-2 counts (Witten–Bell to the unigram) | 2.756 |
| order-3 counts | 2.280 |
| order-4 counts | 2.186 |
| byte-window expert, same body (4 epochs, its best DEV) | 3.178 |
| **chosen expert, Cascade, seeds 1 / 2 / 3** | **2.054 / 2.051 / 2.049 — mean 2.051** |
| float control: every matrix float32 + Adam, activations unquantised, seeds 1 / 2 / 3 | 1.998 / 2.003 / 1.997 — mean 1.999 |

- The Cascade expert beats the best count model by **0.135 bits per byte** and trails float weights by
  **0.052 (2.6%)** — for one byte of state per weight against 12 (a float and two Adam moments).
- Seed-to-seed spread 0.002 — a tenth of the 0.02 decision margin: no decision below rests on noise.
- **Second language** (`corpus_fr.txt`, 80 KB of TRAIN, the configuration unchanged): **2.336** on TEST
  against 2.379 for the best count model (order 3). The margin is smaller on the smaller corpus, and on
  French DEV it is 0.009 -- reported as found.

## The decisions, on DEV

| | arms (best DEV bits/byte) | decision |
|---|---|---|
| Q1 input | **hash 2.181** · window 3.165 | hash, by a full bit |
| Q2 bag normalisation | **per unit 2.181** · per feature 2.382 | per unit |
| Q3 buckets | 1024: 2.181 · **4096: 2.117** · 16384: 2.120 | 4096 (16384: 4x the state, no gain) |
| Q4 orders | 4: 2.233 · **6: 2.117** · 8: 2.184 | 6 (8 overfits from epoch 2) |
| Q5 hidden layers | 1: 2.117 · **0: 2.091** | 0 -- better AND cheaper |
| Q6 head | **Cascade 2.091** · float 2.043 | Cascade: float wins by 0.048, under the 0.05 bar |
| Q7 width | 256: 2.091 · **512: 2.044** | 512 |
| Q8 schedule | cosine 0.3: 2.044 · **constant 0.1: 2.064** · constant 0.3: 2.077 | constant 0.1: the cosine wins by exactly 0.020, and the rule asked for MORE than 0.02 |

What the decisions say:
- **A context feature names a context** -- which is what a memory of text is. Reading hashed contexts
  instead of a window of byte embeddings is worth a full bit per byte with the same Cascade body; the
  window expert is capped near 3.2 by its ternary layers (its float twin reached 2.36 in the exploration
  below).
- **How the Cascade update is normalised decides what it learns.** Per hidden unit, over every feature
  (zeros counted), a feature's step is proportional to the evidence it had in the batch; per feature
  row, a context seen once moves as far as one seen a hundred times (0.2 bits per byte worse).
- Every configuration peaks around epoch 2 and then overfits: the expert memorises its training text
  (train sample 1.52 bits per byte at the chosen configuration's second epoch). For a slow store that
  memorisation is the POINT; the held-out number is what says it also generalises.

## Exploration before the pre-registration -- superseded, kept for the record

The first sweeps (the numbers quoted in the Phase 1 report) chose among designs by the same held-out
slice they reported: the last 10% of the file, which is one whole unseen book (Sherlock Holmes). That is
selection on the test set, and it is why P2.1.1 was redone as above (COMMANDMENTS W-P2.1-1). Their
direction held: hash 2.43 vs window 3.19 then; 2.18 vs 3.17 on DEV now.

## Notes on the harness
- The first eight runs lost their final JSON line to a numpy float that `json` cannot write; their
  records in `runs.jsonl` were rebuilt from the per-epoch progress logs (three decimals) and say so.
- The float control's embedding bag is trained with **sparse Adam** (only the rows a batch touches move;
  global step for the bias correction -- PyTorch's `SparseAdam`). Dense Adam over all 12.6 M floats every
  step took 9 minutes an epoch; a first dense run was stopped after one epoch (DEV 2.049, against 2.088
  for the sparse control's first epoch and 2.018 for its best) and is not in the table.
- The sparse Cascade update used for the bag (`cascade_update_cols`) is the full update restricted to the
  touched columns: proven identical position for position (five steps, 500 x 7, and again at scale: the
  full and sparse versions of the same run gave the same bits per byte to the third decimal).

## Open for later (not decided here)
- Order 1 has only 257 symbols; hashing it into 4096 buckets collides about 8 pairs. An exact order-1
  table is free and could only help -- to be measured, with the same rule, when P2.5 revisits the design.
