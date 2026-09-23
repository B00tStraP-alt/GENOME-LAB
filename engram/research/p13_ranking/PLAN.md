# P1.3 episodic store -- opening measurements (plan, not yet run)

Carried from P1.2:
- retrieval = signature containment (1 KB) -> top C -> exact (streams text). Dense 512 = geometry for slow store.
- encoder deterministic + ~30 MB/s -> dense can be RECOMPUTED from stored text (17 us / 500 B).
- signature saturates on long texts -> episodes must be bounded chunks (ENGRAM_EPI_TAIL 480 B?).
- W-P1.2-10 open item: stage-1 recall@C measured only at N = 860. Must be measured at scale.

Experiments, each with a control, before any design is frozen:
1. SCALE: build a larger public-domain corpus (Gutenberg books in scratchpad: pg11, pg84, pg1342,
   pg1661, pg2229, pg13371, pg14155, pg22367, pg23950, pg24264), chunk to episodes, N up to ~50K.
   Measure stage-1 recall@C vs (N, C) for signature vs dense; cascade r@1; latency per query.
   Decide C(N) policy. Control: exhaustive exact on a subsample.
2. CHUNKING: chunk size (240/480/960 B, sentence-aligned vs fixed) vs fragment recall + signature fill.
3. SPARSE CODE (expand-then-sparsify, E=2048, K=32, 64 B): does it earn a role?
   (a) as first stage vs signature; (b) as near-duplicate / novelty detector for replay (P4.1).
   If it earns nothing measurable: drop it, ledger row. FANIN dial measured only if it earns a role.
4. RESIDENT SET: signature + text resident, dense recomputed -- memory per episode, recall latency.
5. LESION CONTROL: shuffled signatures must collapse recall (proves the signature carries the signal).
6. Proofs: fault injection on every allocation, capacity/eviction, determinism, degraded cues,
   tombstones, id stability, stats accounting.

## Experiment 1 results (explore8, scale_chunks.txt: 21,811 chunks <= 480 B, 5 languages, 300 sources x 12 arms)
N      sig r@10  sig r@50  dense r@10 dense r@50  cascade r@1 C=10 / C=50 / C=500   exhaustive
1000   0.9967    0.9986    0.9719     0.9892      0.9806 / 0.9797 / 0.9797          0.9797
5000   0.9958    0.9981    0.9361     0.9714      0.9619 / 0.9597 / 0.9589          0.9589
21811  0.9883    0.9942    0.8958     0.9419      0.9242 / 0.9189 / 0.9172          (not run)
stage-1 cost: sig 5.6 ms/query at 21.8K (per-doc loop, not bit-sliced); dense 17 ms.
FINDINGS: (a) signature candidate recall holds at scale; dense does not.
(b) cascade r@1 DECLINES with N and with C -> the final exact score is the weak link at scale:
    it has no notion of feature RARITY; containment at small C already out-ranks it.
NEXT: IDF / BM25-style store-level ranking (store statistics are legitimate at the store level,
unlike in the encoder -- W-P1.2-5); bit-sliced signature layout for speed; C policy.
