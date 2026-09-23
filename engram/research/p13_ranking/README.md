# P1.3 — how the episodic store ranks, and the evidence for it

Every number in `src/engram_store.h` ("THE RANKING", "WHY, IN NUMBERS") and in ledger rows W-P1.3-7
to W-P1.3-11 of `COMMANDMENTS.md` comes from the programs in this directory. They are lab code:
run by hand, not part of the build or the gate. The gate re-measures the final design through the
store's own API in `test/test_store.c` S8.

## The question

At 21,811 episodes the P1.2 cascade got worse as it looked at more candidates. Stage 1 (the
signature) still held the source for 99.5% of cues at C = 50. Stage 2 (the Bhattacharyya score)
ranked wrongly: exhaustive exact search, 0.911, was worse than a 10-candidate cascade, 0.929
(`results/base_dev.json`). The question was what the final score should be.

## The protocol (fixed before any approach was tried: `lab.c`)

* **Store:** the first N chunks of `corpus_scale.txt` (21,811 chunks of ≤ 480 bytes, five
  languages, shuffled). `make_scale_chunks.py` makes it from the ten Project Gutenberg texts that
  `tools/build_corpus.py` uses; SHA-256
  `ec02a382e1d0a452555884cffb81c118387008fd217e5e70afb1ddf1cf3dbec7`.
* **Sources:** chunk `(t * 7919 + 13) mod N`. DEV is even t and TEST is odd t, 300 sources each.
  All choices are made on DEV. TEST is run once per decision.
* **Cues:** 12 per source, 5 / 8 / 10 / 15% fragments × 0 / 1 / 2 typos (the test_enc
  generators).
* **Control:** `rarity4.c` builds keyword cues (3 / 5 / 8 pieces of the source in random order)
  and a shuffled 15% window. They have no contiguous text, so they are the cue a
  word-order-sensitive score could hurt.
* **Ties:** from `rarity2.c` on, every score reports opt (ties won) / exp (ties split) / pess
  (ties lost). `rarity.c` counted ties as wins, and that is how IDF appeared to help
  (W-P1.3-8). Its output is kept, in `results/rarity_dev.txt`, as the record of the mistake.

## The sequence

| program | asks | answer (DEV, C = 50) |
|---|---|---|
| `explore8.c` | does stage 1 hold up at scale? | signature recall@50 0.994 at 21.8 K (dense 0.942); the final score is the weak link |
| `rarity.c` | IDF / BM25 as the final score? | containment-IDF "0.963", counting ties as wins |
| `rarity2.c` | the same, ties counted honestly; where df comes from | plain containment 0.949 vs containment-IDF 0.947 (ties split): IDF earns nothing. Integer-log IDF equals float IDF to 5.5e-6 nats |
| `amb.py` | how much is left on no-typo cues? | no-typo cues only: ceiling 0.982 (some cues' exact text also sits in another chunk); lex(containment-IDF, exact) reaches 0.980 |
| `rarity3.c` | does alignment help? | lex(−edits, containment-IDF, exact) 0.977 vs lex(containment-IDF, exact) 0.953 (plain containment: 0.956) |
| `rarity4.c` | the control | ungated alignment collapses keyword cues to 0.52: rejected |
| `dump.c` + `evalrules.py` / `gate2.py` / `discord.py` / `gap.py` | which rule is safe on both? | the significance gate (edits ≤ median/3, ≤ \|q\|/3, containment within 0.2 of the best): fragments 0.976, control unchanged |
| `preregistered.txt` | — | the rule, written down before TEST was run |
| TEST (`results/*_test_*`) | confirmation, once | fragments +40 −1 paired, control +0 −1 (one ambiguous 3-word cue, examined by hand) |
| `e2e.c` | through the real store API, on a protocol never seen | see below |

## End to end (`e2e.c`)

Whole paragraphs of `test/data/scale_docs.txt` go through `engram_store_add` (26,139 episodes).
Cues of 12 / 16 / 24 / 40 / 64 / 96 bytes are cut at any word start with 0 / 1 / 2 typos, so a cue
may straddle two episodes. One recall with k = C gives every candidate. The three rankings are
compared on the same candidates.

| | fragments DEV | fragments TEST | control DEV | control TEST |
|---|---|---|---|---|
| EXACT (P1.2) | 0.793 | 0.789 | 0.869 | 0.876 |
| BAG | 0.889 | 0.888 | 0.911 | 0.907 |
| FULL | **0.922** | **0.918** | 0.910 | 0.908 |
| FULL vs BAG, paired | +182 −0 | +163 −0 | +1 −3 | +1 −0 |

**Chunk overlap:** `e2e_dev_ov*.txt` uses the first protocol's four lengths (7th argument `4`).
Overlap 64 or 128 bought 2 or 5 cues out of 3,600, for 3% or 9% more text. The default stays 0.
The six-length re-run, which includes the short cues (`e2e6_dev.txt`, `e2e6_dev_ov64.txt`,
`e2e6_dev_ov128.txt`), gives fragments 0.9222 / 0.9228 / 0.9226: no gain there either. The committed
`e2e.c` reproduces `e2e6_dev.txt` byte for byte. Its four-length output reproduces
`e2e_dev_ov0.txt` too, plus the paired-count line that was added later.

## Reproduce

```sh
python3 make_scale_chunks.py            # in a directory holding pg11.txt ... pg24264.txt -> corpus_scale.txt
./build.sh rarity2.c rarity2 && ENGRAM_TEST_DATA=. ./rarity2 21811 dev 300
./build.sh dump.c dump && ENGRAM_TEST_DATA=. ./dump 21811 dev 300 frag && ENGRAM_TEST_DATA=. ./dump 21811 dev 300 ctrl
python3 evalrules.py dev 50; python3 gate2.py dev 50
gcc -O2 -std=c99 -ffp-contract=off -I../../src -o e2e e2e.c ../../src/*.c -lm -lpthread
./e2e ../../test/data/scale_docs.txt 0 300 dev        # add "50 4" for the four-length protocol
```

The encoder, alignment and store are deterministic (R5), so a re-run gives the same numbers,
except for timings.
