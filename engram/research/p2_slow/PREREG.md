# P2.1.1 — pre-registration (written before any DEV run)

## The data
`corpus_en.txt`, paragraphs interleaved: i % 10 == 8 → DEV, == 9 → TEST, else TRAIN (expert.py `split`).
Confirmation on a second language: `corpus_fr.txt`, same rule.

## The measure
Bits per byte on DEV, the best of 4 epochs, seed 1. Batch 128, the P2.1 body (expert.py docstring).

## The questions, decided in this order, each fixing the winners before it
| | question | arms |
|---|---|---|
| Q1 | what the expert reads | `hash` (6 orders x 1024) vs `window` (12 bytes x 16-d embeddings) |
| Q2 | how the bag's Cascade update is normalised | per hidden `unit` (over all F features) vs per `feature` row |
| Q3 | buckets per order | 1024 vs 4096 vs 16384 |
| Q4 | orders | 4 vs 6 vs 8 |
| Q5 | hidden H x H layers | 0 vs 1 |
| Q6 | head | Cascade vs float |
| Q7 | width H | 256 vs 512 |
| Q8 | learning-rate schedule | cosine 0.3 vs constant 0.3 vs constant 0.1 |

**Decision rule.** The arm with the lowest DEV bits per byte wins, unless its margin over a cheaper arm
(fewer bytes of state, or less work per byte) is under 0.02 bits per byte — then the cheaper arm wins.
0.02 is fixed now; the seed-to-seed spread measured on the final configuration is reported beside it,
and if that spread turns out larger than 0.02 the affected decisions are re-examined in the open.

Q6 (float head) is also judged against the project's principle — every weight a Cascade weight, no
float per weight: a float head must win by more than 0.05 to be adopted (it is 256 x H floats of state
and an Adam per weight).

Q8 is judged for what the slow store needs: a memory never "finishes" training, so a constant rate is
preferred unless the cosine wins by more than 0.02.

## What TEST is used for — once, after every decision
1. The chosen configuration, seeds 1, 2, 3.
2. Its float control (`--float`: every matrix float32 + Adam, activations unquantised), seeds 1, 2, 3.
3. The `window` arm with the chosen body, seed 1.
4. The count models (unigram; orders 1–4, Witten–Bell interpolation to the unigram).
5. The same, chosen configuration and count models on `corpus_fr.txt`.

Nothing seen on TEST changes a decision. If TEST contradicts DEV, that is reported as found.
