# ENGRAM — Roadmap

ENGRAM is a private memory for a person's computer that **learns into its own weights while the machine
is idle** — and can prove, for every single thing it learned, that learning it broke nothing it already
knew. Phase 1 built the ground it stands on. Phases 2–6 build the mind, its conscience, its sleep, its
armour, and its door to the world.

Phase 1 was organised as 5 sub-phases. **From Phase 2 on, every sub-phase has three mini-phases**
(P2.1.1, P2.1.2, P2.1.3, …) — 75 mini-phases in all — and every one of them ends with evidence that
could have failed (COMMANDMENTS R6).

---

## What would make it a breakthrough — the claims, and how each will be proven

A claim is written here before it is measured; it is kept only if the measurement says so, with its
control beside it (R4). None of these is claimed yet.

| # | Claim | Proven in | Control |
|---|---|---|---|
| B1 | **Bit-identical learning.** The same memories, learned on Linux or Windows, on 1 thread or 8, give the same weights to the last bit — a model anyone can re-derive and audit. | P2.2.3, P2.5.3 | the float-BLAS training every framework does, which does not |
| B2 | **Six bits hold a weight's whole training state** — no float shadow, no optimiser moments — and forget less than float-metaplastic training when learning never stops. | P2.5.2 | κ = 0 (no depth plane); float weights + Adam |
| B3 | **Every update is a transaction with a two-sided verdict**: it must improve what it was meant to learn AND not measurably regress what was already known — or it is rolled back to the bit. | P3.2, P3.5 | the same updates applied without a verdict |
| B4 | **Mitosis instead of forgetting.** When new knowledge truly conflicts with old, the expert divides rather than overwrite. | P3.4 | the same stream with division disabled |
| B5 | **Knowledge moves from memory into weights** while the machine sleeps, measurably: after consolidation the weights answer what the episodic store has since evicted. | P4.3, P4.5 | Z: no consolidation; R: naive fine-tuning at the same budget |
| B6 | **Custody of knowledge.** A signed, hash-chained diary of everything learned, every verdict and every weight bit moved; any edit to diary or weights is detected. | P4.4 | a planted edit at every position |
| B7 | **A low-end laptop is enough.** Sleep cycles, memory and latency budgets met on a 2-core, 4 GB machine; everything encrypted at rest; no network, ever. | P5.5 | the budgets, stated before measuring |
| B8 | **Any assistant can remember.** An MCP server gives any AI assistant a private, learning, auditable long-term memory. | P6.2 | a real MCP client, end to end |

---

## How every mini-phase is run

* **.1** is usually *research or design*: a measured question, answered with controls on a development
  split and confirmed once on a held-out split; or a written contract with its threat model.
* **.2** is usually *the build*: the code, cross-checked where possible against an independent
  implementation (Python references, RFC vectors), with its suite.
* **.3** is usually *the proof*: adversarial tests, fault sweeps, mutation, cross-platform identity —
  and the sub-phase's own gate.

Each mini-phase ends with the strict Linux build and every suite green; each **.3** ends with the full
gate (`make gate`: Linux, ASan+UBSan, TSan, Windows build, Wine, cross-platform identity, system DLLs
only). Each phase ends with its report to the owner. A mini-phase is committed and pushed when its
evidence is in, and not before; below, ✅ and a commit mark each one that is. Work stops at the end of
each sub-phase until the owner says to go on.

---

## Phase 1 — the ground (done)

| | | |
|---|---|---|
| P1.1 | Core runtime: checked allocator with fault injection, atomic files, platform layer | ✅ |
| P1.2 | Encoder: UTF-8 text to vectors and signatures, identical on every platform | ✅ |
| P1.3 | Episodic store: bounded episodes, recall at scale (0.92 fragment recall@1, 26K episodes) | ✅ |
| P1.4 | Router: IVF over unit keys (and the measurement of what it cannot do) | ✅ |
| P1.5 | Crypto, sealed container, keyfile, persistence; the Windows EXE gate | ✅ |

---

## Phase 2 — the slow store: weights that learn (in progress: P2.1 and P2.2.1 done; next P2.2.2)

**P2.1 — The expert** ✅ gated
- **P2.1.1** ✅ `9a49b39` Research: what the expert reads and how its weights move. Hashed contexts vs a byte
  window; normalisation orientation of the Cascade update; head precision; width, buckets, orders;
  constant vs scheduled learning rate. Chosen on a DEV split, confirmed once on TEST, against count
  models and a float control. Deliverable: `research/p2_slow/`.
- **P2.1.2** ✅ `fdfe6a4` Build the primitives: `engram_math` (exp/log2 from + − × ÷), `engram_cascade` (the word,
  the update, the integer kernels, persistence), `engram_ctx` (hashed contexts). Each bit-exact against
  an independent Python reference; MATHPRINT/CASCADEPRINT across platforms.
- **P2.1.3** ✅ `6b58892` Build the expert's forward pass and `bits()`: bit-exact against the Python reference;
  allocation sweeps; EXPERTPRINT across platforms; speed measured. Sub-phase gate.

**P2.2 — Learning**
- **P2.2.1** ✅ `63d1c6e` The backward pass: norm, straight-through quantisation, bag; gradients checked against a
  float finite-difference shadow.
- **P2.2.2** The training step: Cascade updates (rows and units) + Adam for gains and bias; N steps
  bit-exact against the Python reference.
- **P2.2.3** Deterministic parallelism: work split by output element; 1 thread = N threads to the
  bit; TSan clean; speed-up measured; TRAINPRINT Linux = Windows (**B1**). Sub-phase gate.

**P2.3 — What "better" means**
- **P2.3.1** The cost as a function pointer: byte cross-entropy, masks, per-position weights; the same
  function trains and judges (R7).
- **P2.3.2** Scorers the verdict will need: familiarity (bits under the expert vs a background model),
  paired per-position cost differences.
- **P2.3.3** Proof: gradient checks per cost; a planted wrong cost caught. Sub-phase gate.

**P2.4 — Reachable experts**
- **P2.4.1** Keys: an expert's key is the centroid of what it learned; incremental update; key quality
  measured.
- **P2.4.2** The slow store: experts + the P1.4 router over their keys; save/load sealed
  (`ENGRAM_KIND_SLOW`).
- **P2.4.3** Proof: experts for five languages, a cue reaches its own; round trips; fault sweeps.
  Sub-phase gate.

**P2.5 — Gate: the slow store demonstrably learns**
- **P2.5.1** Learning against the bars: held-out bits per byte vs unigram, order-k counts, the
  untrained expert; trained text vs unseen text (the weights hold what they were shown).
- **P2.5.2** Learning without end: languages in sequence, κ vs κ = 0 vs float (**B2**).
- **P2.5.3** Phase gate, mutation campaign, Phase 2 report.

## Phase 3 — the transaction: every update is a verdict

**P3.1 — Snapshot and bit-exact restore**
- **P3.1.1** Design: copy-on-write of the rows an update touches vs a full copy — cost measured.
- **P3.1.2** Build `engram_txn`: begin, commit, abort; restore to the bit (fingerprint).
- **P3.1.3** Proof: allocation and IO failure at every point inside a transaction leaves exactly the
  old state; random abort points fuzzed. Sub-phase gate.

**P3.2 — The two-sided verdict**
- **P3.2.1** Research: the statistics — paired per-position bits on the target and on a guard of what
  is already known; a deterministic resampling test; thresholds chosen on dev, error rates measured.
- **P3.2.2** Build the verdict engine on the P2.3 cost.
- **P3.2.3** Proof: planted regressions refused, planted improvements accepted, false-accept and
  false-refuse rates reported. Sub-phase gate.

**P3.3 — Bit-level accounting**
- **P3.3.1** The model: per transaction, which positions moved and which values flipped, per matrix
  and row; state bits changed.
- **P3.3.2** Build the accounting record, compact and hashed.
- **P3.3.3** Proof: the record equals an independent diff of the two snapshots, exactly, on every
  platform. Sub-phase gate.

**P3.4 — Mitosis**
- **P3.4.1** Research: what evidence says "conflict" (the target only improves by regressing the guard);
  how a child is born (clone, key split).
- **P3.4.2** Build division: clone, train the child on the new material, split the keys, update the
  router.
- **P3.4.3** Proof: planted contradictions a single expert cannot hold are held by two; growth bounded
  (**B4**). Sub-phase gate.

**P3.5 — Gate: rollback proven, regression refused**
- **P3.5.1** Rollback at scale: thousands of transactions, random aborts, fingerprints checked.
- **P3.5.2** End to end: a planted harmful batch is refused; the same batch without the verdict does
  the harm (**B3**).
- **P3.5.3** Phase gate, mutation campaign, Phase 3 report.

## Phase 4 — sleep: consolidation while the machine is idle

**P4.1 — What to replay**
- **P4.1.1** Research: order of consolidation — retrieval-ordered, recent, surprising (bits under the
  slow store); measured.
- **P4.1.2** Build the replay queue and the guard sampler over the episodic store.
- **P4.1.3** Proof: deterministic, bounded, fair. Sub-phase gate.

**P4.2 — When to sleep**
- **P4.2.1** Idle and power: Windows input idle, battery, mains; the budget policy.
- **P4.2.2** Build the scheduler: work in transaction-sized slices, yield the moment the user returns.
- **P4.2.3** Proof: preemption at every point leaves a consistent state; simulated clocks. Sub-phase
  gate.

**P4.3 — The consolidation loop**
- **P4.3.1** Build: replay → transaction → verdict → commit / abort / divide → mark consolidated.
- **P4.3.2** Measure: after consolidation, the weights answer what the episodic store has evicted
  (**B5**).
- **P4.3.3** Proof: a crash at any point resumes exactly once. Sub-phase gate.

**P4.4 — The custody diary**
- **P4.4.1** Design: hash-chained, authenticated records of every transaction: what, verdict, bits
  moved, fingerprints before and after.
- **P4.4.2** Build the append-only diary, its verifier and a human-readable rendering.
- **P4.4.3** Proof: every edit, truncation or reordering detected; diary and weights cross-checked
  (**B6**). Sub-phase gate.

**P4.5 — Gate: arms A / Z / R at identical budget**
- **P4.5.1** Pre-registered protocol: A = ENGRAM, Z = no consolidation, R = naive fine-tuning.
- **P4.5.2** The runs, on held-out material.
- **P4.5.3** Phase gate, mutation campaign, Phase 4 report.

## Phase 5 — armour

**P5.1 — Encryption at rest, everywhere**
- **P5.1.1** Threat model: what an attacker with the disk, or with a copy of it, can and cannot learn.
- **P5.1.2** Wire the sealed container through every file ENGRAM writes.
- **P5.1.3** Proof: no plaintext of any stored memory anywhere on disk (scanned). Sub-phase gate.

**P5.2 — Crash safety**
- **P5.2.1** Design: exactly-once batches; the recovery journal.
- **P5.2.2** Build recovery on start.
- **P5.2.3** Proof: power loss simulated at every IO operation of a whole sleep cycle. Sub-phase gate.

**P5.3 — System-wide fault injection**
- **P5.3.1** Every allocation and IO operation of whole workflows enumerated.
- **P5.3.2** Sweeps: each failed in turn; state checked after each.
- **P5.3.3** Proof and report. Sub-phase gate.

**P5.4 — Fuzzing every loader**
- **P5.4.1** Harness: coverage-guided, deterministic seeds, every file kind.
- **P5.4.2** Campaigns under ASan/UBSan.
- **P5.4.3** Every finding fixed, its input kept as a regression test. Sub-phase gate.

**P5.5 — Gate: a low-end laptop is enough**
- **P5.5.1** Budgets written before measuring: memory, sleep-cycle time, recall latency.
- **P5.5.2** Measured on a constrained machine (2 cores, 4 GB).
- **P5.5.3** Phase gate, Phase 5 report (**B7**).

## Phase 6 — the door to the world

**P6.1 — The command line**: **.1** the interface (remember, recall, sleep, status, diary); **.2** build;
**.3** end-to-end tests and gate.

**P6.2 — The MCP server over stdio**: **.1** the protocol surface; **.2** build; **.3** conformance with a
real MCP client and gate (**B8**).

**P6.3 — The Windows app**: **.1** design of the tray app and the diary view; **.2** build (Win32, no
dependencies); **.3** tests under Wine and gate.

**P6.4 — Packaging**: **.1** the single EXE, manifest, icon, version resource; **.2** build and
reproducible-build check; **.3** clean-machine install test and gate.

**P6.5 — Release**: **.1** documentation; **.2** end to end under Wine from a clean prefix; **.3**
release candidate, final gate, final report.
