# P1.4 — what an IVF router can do in ENGRAM's embedding space, and what it can't

Every number in `src/engram_router.h` about recall and cost, and in ledger rows W-P1.4-*, comes
from `ivf.c` here or from `test/test_router.c` R7/R8 in the gate.

## The question

The router is an inverted-file index: spherical k-means centroids, one bucket per key, and a search
that scores only the `nprobe` nearest buckets. It was specified for the slow store's expert keys
(P2.4). P1.3 left a second question open: could it make the episodic store's stage 1 sub-linear?
Stage 1 scans every signature: 11 ms at 26 K episodes, linear in the store.

## The measurement (`ivf.c`)

Setup:

* **Store:** 21,811 chunks of `corpus_scale.txt` (see `../p13_ranking/`).
* **Cues:** the 3,600 dev fragment cues of the P1.3 lab.
* **Keys:** dense unit vectors (`engram_encode`).

Two recalls, for C = 64 / 148 (√N) / 512:

* **ANN:** of the exhaustive dense top 10, how many the IVF top 10 holds.
* **Store:** whether the source still reaches the signature top 50 when only probed buckets are
  scanned. Full scan: 0.9953.

| C | nprobe | keys scanned | ANN recall@10 | store: source in top 50 |
|---|---|---|---|---|
| 64 | 8 | 13% | 0.743 | 0.723 |
| 64 | 32 | 51% | 0.972 | 0.955 |
| 148 | 8 | 5.8% | 0.577 | 0.580 |
| 148 | 64 | 44% | 0.967 | 0.950 |
| 512 | 64 | 14% | 0.827 | 0.823 |

(`results/ivf_dev.txt`)

## What it means

For fragment cues this space has almost no bucket structure an IVF can use. A fragment's dense
vector keeps a fraction of its source's mass. Its "exhaustive top 10" sit at cosines barely above
the rest, and they are spread across the whole sphere. Holding the source for 95% of cues takes
scanning about half the store, against 99.5% for the full signature scan. So the episodic store
keeps its full scan (W-P1.4-1).

A cluster-pruned exact search was also considered and rejected. It would bound each bucket's best
possible cosine by cos(φ − θ): φ is the query's angle to the centroid, θ the bucket's widest member.
But members sit at cosines of 0.3–0.5 from their centroids, so every bound comes out near 1 and
nothing is pruned.

The router is kept for its specified job, routing to expert keys. There a cue sits close to the
key it should reach, which is the near-copy regime that `test_router` R7 measures. Beside the
approximate search it has an exact search, which is the right choice while experts number in the
thousands.

## Reproduce

```sh
cp ../p13_ranking/corpus_scale.txt .     # see ../p13_ranking/make_scale_chunks.py
../p13_ranking/build.sh ivf.c ivf && ENGRAM_TEST_DATA=. ./ivf 21811 300
```
