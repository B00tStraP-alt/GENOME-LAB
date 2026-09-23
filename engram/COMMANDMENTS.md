# ENGRAM — Commandments

These bind every line of this tree. They are recited before a phase is gated, and a phase that
violates one is not done.

---

## How the work is done

**I. Never fan out.** Every task is done by the builder directly. No delegation, no sub-agents, no
parallel orchestration. One mind holds the whole design, so nothing is lost at a hand-off.
This holds whatever mode the tooling is in: in P1.3 a multi-agent workflow was launched because a
harness mode ("ultracode") asked for one, and the owner stopped it — "dont fan out". No agent's output
from that run is used; the work it was doing was done again by the builder.

**II. The owner's direction governs what is built. Evidence governs what is claimed.**
What to build, what to prioritise, and what matters is the owner's call and is followed without
argument. What a measurement *says* is not anyone's call: it is reported exactly as it came out,
including when it contradicts what was hoped. The two never conflict — honest evidence is what makes
the owner's direction worth following.

**III. Never be conservative at a roadblock. Find the way through.** A missing tool is installed. A
failing dependency is replaced. "It can't be done here" is the start of the work, not the end of it.

**IV. Attention to detail. Root-cause every issue.** A symptom is not fixed until its cause is named
and removed. A fix that makes a test pass without explaining why the test failed is a second defect.

**V. Phase gating.** No phase begins until the previous one is fully built, bulletproof,
smoke-tested, and actually functioning. Each phase has five sub-phases and each is solidified before
the next. Integration happens only when every phase underneath it stands on its own.

---

## When there is a wall

**VI. A ceiling is never accepted, and never painted over.**
When a problem cannot be overcome with the obvious tools — a ceiling, a wall, a constraint — the
answer is a novel solution, arrived at deliberately:

1. **Research it.** Read what the field actually knows: papers, primary sources, the mechanism behind
   the tool that already solved a cousin of this problem.
2. **Use the ingredients.** The VECTRA, VX, TRACE and GenomeLab designs are a library of solved hard
   problems. A wall here is often a solved problem there.
3. **Combine.** Research with architectural change. A novel solution with another novel solution.
   Knowledge with measurement.
4. **Measure the way through.** A solution is not real until an experiment with a control says so.

**VII. Never paper over. Never lie. Never fake it. Never hallucinate.**
- A number that was not measured is not written down.
- A capability that was not demonstrated is not claimed.
- A gap is named as a gap, with the attack on it written beside it.
- "It works" means it ran, under test, and the test could have failed.
- A source that was not read is not cited. A function that was not verified is not relied on.

---

## Engineering rules

**R1 — Nothing that can fail is void.** It returns an `engram_rc`. The sticky-error buffers are the
one documented shape that defers the report, and the only way to take data out of them returns the rc.

**R2 — Every allocation goes through `engram_alloc`.** So every one of them can be failed on
purpose, and every failure path is exercised by a test rather than trusted.

**R3 — Every file is written atomically.** A crash leaves the old complete file or the new complete
file, never a torn one.

**R4 — Nothing is measured without its control.** A number that could not have come out differently
is not a measurement. Every comparison carries an arm that prices the mechanism itself.

**R5 — Determinism.** The same inputs produce the same bytes on every run, build and machine.

**R6 — An instrument that cannot fail is not an instrument.** Every gate is proven capable of
failing by planting a defect it must reject. (First enforced in P1.1, where the compile gate was
caught printing PASS regardless of the compiler's exit code.)

**R7 — Two vectors compared must be built by the same function.** Comparing a query and a key built
by different code is the ROUTE-SPACE defect: nothing crashes, every counter stays green, and the
answer is about a different question.

**R8 — A sentinel is never a bar.** `0.0` meaning "not measured" must never be compared as if it
were a measurement.

---

## The ledger of walls met, and how each was passed

Rows are numbered `W-<phase>-<n>` so code and tests can cite the row that explains them.

| Phase | Wall | Research | Way through |
|---|---|---|---|
| W-P1.1-1 | A magic number cannot detect double-free on glibc: tcache overwrites the first 16 bytes of a freed block, and a reused block reads LIVE again | AddressSanitizer's quarantine allocator | Poison + DEAD-mark + FIFO quarantine with a byte budget; honest limit documented |
| W-P1.1-2 | The compile gate printed PASS even when gcc failed (`head` returned 0) | — | Gate rewritten on real exit codes; proven by a planted defect it must reject (R6) |
| W-P1.1-3 | `"????-??-??T..."` contains `??-`, a live C99 trigraph for `~` | C99 §5.2.1.1 | Placeholder rewritten as `XXXX-XX-XX...` -- also no longer resembles a real timestamp |
| W-P1.1-4 | `log()` is not required to be correctly rounded; glibc and the Windows CRT can differ in the last bit, so a Box-Muller initialiser would differ per OS | IEEE-754 guarantees `sqrt`, not transcendentals | No libm transcendentals in anything reproducible: Irwin-Hall normal, uniform init |
| W-P1.1-5 | FMA contraction lets `a*b+c` round once on one CPU and twice on another | GCC `-ffp-contract` semantics | `-ffp-contract=off` in every build variant |
| W-P1.1-6 | The stray-temp test probed `<path>.tmp.0.N` -- the PID is never 0, so it could never find anything | -- | Real directory listing; the same code became the orphan sweep that cleans up after a KILLED process |
| W-P1.1-7 | Pinned timestamps were typed, not computed | -- | Every one verified against Python `datetime`, an independent implementation |
| W-P1.1-8 | Tests that plant corruption on purpose would trip the "corruption must be zero" invariant | -- | Planted defects are DECLARED and the invariant requires counts to equal exactly what was planted; one more fails |
| W-P1.1-9 | LeakSanitizer reported the block the allocator abandoned on purpose | LSan interface | The allocator names the one pointer it abandons (`__lsan_ignore_object`); LSan stays live for all others |
| W-P1.1-10 | `engram_now_ns` on Windows cached its frequency in an unlocked static, excused as "a benign race" | C11 memory model: every data race is UB | Shared state deleted; the constant is queried per call |
| W-P1.2-1 | The Chinese source text is hard-wrapped, and a legacy licence footer sits INSIDE the body, so naive paragraph splitting produced fragments and boilerplate | Inspection of the raw Gutenberg files | Paragraphs assembled by indentation and joined without spaces; the footer stripped by its marker; ASCII spaces in the original verified present rather than assumed |
| W-P1.2-2 | The maximum compatibility-fold expansion was assumed to be 3 codepoints | Measured over all of Unicode 14.0 | It is 4 ("⑽" → "(10)"); `ENGRAM_FOLD_MAX` generated from the measurement, never typed |
| W-P1.2-3 | A full per-codepoint reference file for the Unicode tables was over 1 MB | -- | Run-length classes plus only the non-identity folds: 38 KB, still exhaustive over the BMP |
| W-P1.2-4 | Generated table names collided with enum constants | -- | Renamed in the GENERATOR (`ENGRAM_FOLDTAB_*`); generated output is never hand-edited |
| W-P1.2-5 | Mean-centring (All-but-the-Top, Mu & Viswanath 2018) cut the unrelated-pair cosine 0.23 → 0.04 -- and made retrieval WORSE (5% fragments 0.40 → 0.28) | Isotropy post-processing literature | Rejected: a better-looking metric bought a worse answer on the real task, and would have made vectors depend on corpus statistics |
| W-P1.2-6 | The encoder suite's ablation (T7) and dimension (T8) gates ran on whole paragraphs, where EVERY arm scored recall@1 = 1.0000: the gates could not fail (R6) | -- | A fragment objective (5-15% of a paragraph, 0-2 typos) on which the default scores well under 1; a saturation guard that fails the suite if the default ever reaches 0.98; every "must not gain" gate refuses to certify from a saturated instrument |
| W-P1.2-7 | Short fragments found their source only 40% of the time; the best WRONG paragraph out-scored the right one on average | Burstiness (Jégou, Douze & Schmid, CVPR 2009); signed power normalisation (Perronnin, Sánchez & Mensink, ECCV 2010) | Signed square-root TF shaping, built from `sqrt` only (correctly rounded, so R5 holds): 5% fragments 0.40 → 0.58 |
| W-P1.2-8 | No single weighting served both alphabets and Chinese: characters-only was best on four Latin languages and collapsed Chinese from 0.96 to 0.81 | Character n-grams for European retrieval (McNamee & Mayfield, Inf. Retrieval 7, 2004); overlapping-bigram CJK retrieval | The script split: character n-grams for alphabets, ideograph unigrams/bigrams with their own weights; each confirmed on held-out paragraphs of all five languages |
| W-P1.2-9 | Dense recall on short queries kept climbing with D (development runs: 0.79 at 512, 0.93 at 4096, 0.94 at 65536): the loss was hash-collision noise, which falls only like 1/D, and 64 KB per memory is unaffordable | The Hellinger kernel (RootSIFT, Arandjelović & Zisserman 2012) | Under square-root shaping a collision-free vector's squared norm is the text's total feature weight, so the EXACT cosine (the Bhattacharyya coefficient) streams with memory bounded by the query: a zero-allocation exact stage, checked against an independent reference to 1e-12 |
| W-P1.2-10 | The first cascade (dense top 50 → exact) lost to exhaustive exact search by 0.012, beyond its 0.004 margin: T8 FAILED. A cosine asks "how similar overall", a four-word query needs "how much of me does this contain" | Bloom 1970; signature files (Faloutsos & Christodoulakis, TOIS 1984); bit-sliced signatures (BitFunnel, Goodwin et al., SIGIR 2017) | A 1 KB Bloom signature per memory as the first stage: the source reaches the exact stage for 99.5% of queries (dense: 97.1%) and the cascade scores at or above exhaustive exact search in all five languages. No false negatives, proven on 3,694 texts |
| W-P1.2-11 | The first run of the rewritten suite reported a cascade "recall" of 1.42 | -- | `engram_array` does not clear memory; skipped queries left garbage in the hit array. Hits are now zeroed at birth, and EVERY result is checked to hold only 0s and 1s -- an instrument that can say 1.42 can also say a plausible 0.93 that is equally wrong |
| W-P1.2-12 | With the cascade in place, T7 and T12 FAILED on dense-only recall: dropping 3-grams gained +0.033, and quarter-power beat square root by +0.0099 (margin 0.0061). The quarter-power edge had been visible in the development grid and was waved off as noise without a paired test | -- | Selection re-run on the development split with the right objective (the cascade users get), functional constraints first: 3-gram weight 1 → 1/16 (0 would leave one-letter text with no features); square root kept, tied on the cascade and the only shaping whose exact norm streams. Every comparison now paired, clustered by paragraph. Dense-only differences stay printed, marked "reported, not gated" |
| W-P1.2-13 | Encoding ran at 22 MB/s | callgrind | Ten non-inlined finaliser calls and two 64-bit divisions per character, three calls per byte in the iterator. The 4-gram id now chains off the previous 3-gram's (the finaliser is a bijection, so collisions stay 2^-64); one mix gives dimension and sign by multiply-shift; the finaliser is inline; ASCII takes a fast path. 25-35 MB/s: the rest is ~300 instructions spread over the pipeline, and the budget belongs to the P5.5 low-end performance gate |
| W-P1.2-14 | Under Wine, test_core ran 584 checks to Linux's 585 | -- | NOT a defect, recorded so it is not re-investigated: a Linux atomic write has five IO steps (the last, the directory fsync after the rename) and Windows four (`MoveFileExW` with write-through is the commit); the fault sweep injects one failure per step |
| W-P1.2-15 | `make ... \| head; echo $?` reported success for a build that had failed -- the P1.1 gate defect, recurring at the keyboard | -- | `set -o pipefail` and `${PIPESTATUS[0]}` in every interactive build command; the gate already judges real exit codes |
| W-R1-1 | Research, the Cascade weight: three training processes each launching four BLAS threads on four cores made an epoch take 100 s instead of 5 | -- | One BLAS thread per process, four processes in parallel; recorded in the research README |
| W-R1-2 | The first Cascade update capped every move at one position per step, which silently turned large gradients into sign-SGD and penalised the finer-grained words most | Stochastic rounding (unbiased in expectation) | floor(|u|) + one more with probability frac(|u|): expected move equals the gradient step exactly, verified numerically (2.6837 vs 2.683); every Cascade arm re-run, superseded runs discarded |
| W-R1-3 | The ONNX export scored 10% (chance) while the harness had measured 98.02% for the same model; ONNX and numpy agreed with each other exactly, because both read the same wrong input | -- | The save path matched `arm == "cascade"` and so stored decoded values instead of positions for `cascade16`. Fixed to a prefix match; retraining seed 1 reproduced the measured model bit for bit, and the export then matched the harness exactly. Cross-checking against the independently measured accuracy is what exposed it |
| W-R1-4 | Three selected settings sat at the edge of their grids (a step rate of 1.0; κ = 1; κ = 2 for the 4-bit word) | -- | Grids extended until each optimum was interior; the continual-learning result changed materially (κ = 0.5: 0.945 on dev, above every arm) |
| W-P1.3-1 | The episode cutter looked ahead for a paragraph break without bound: a text with none was rescanned from every cut, quadratic in its length | -- | The lookahead stops at the window it could cut in; hostile inputs (all spaces, all newlines, no breaks at all) in test_chunk |
| W-P1.3-2 | A full store that evicts on every add never took the pre-add compaction, so tombstones and their text grew without bound | -- | After the commit (still allocation-free, logically invisible), compaction whenever tombstones exceed a quarter of the live episodes plus 64; S6 runs 400 evicting adds and checks the bound on every one |
| W-P1.3-3 | Eviction cleared LIVE but left CONSOLIDATED on the tombstone, so the consolidated count drifted | -- | Both flags cleared, the signature zeroed; counts cross-checked against the fingerprint in every sweep |
| W-P1.3-4 | The recall scratch was a `double` array reused to hold slot numbers (type punning: undefined behaviour under strict aliasing) | C99 §6.5 ¶7 | A typed candidate record (`engram_cand`, `engram_ranked`) |
| W-P1.3-5 | Store tests failed for reasons in the TESTS: paragraphs over 480 bytes became several episodes and broke the id arithmetic; "!!! ..." is not featureless (punctuation n-grams); a deleted-episode check queried with the wrong text; and the eviction allocation sweep found ZERO allocation points, so it proved nothing (R6) | -- | Count-sensitive tests use one-episode paragraphs; the featureless probe is made of IGNORE characters; the query is the episode's own stored text; every sweep refuses to pass with no allocation points, and the evicting state is built so the add must grow |
| W-P1.3-6 | A multi-agent workflow was launched because a harness mode asked for one; the owner stopped it ("dont fan out") | Commandment I | Stopped, verified no process left; none of its output is used; every measurement below was redone by the builder |
| W-P1.3-7 | At 21,811 episodes the P1.2 cascade got WORSE as it looked at more candidates (C = 10: 0.929, C = 200: 0.912), and exhaustive exact search (0.911) was worse than C = 10: the final score, not the candidate stage, ranked wrong at scale | -- | A store-level ranking, below; the signature stage kept (it held the source for 99.5% of cues at C = 50) |
| W-P1.3-8 | IDF-weighted containment appeared to beat the exact score by +0.034 -- but the lab counted a tie as a win, and containment TIES by construction (every candidate holding the whole cue scores exactly 1). Counted honestly (ties lost, or split), IDF added nothing to plain containment | Robertson & Spärck Jones 1976 (IDF); Robertson & Zaragoza 2009 (BM25) | Every score now reported three ways (ties won / split / lost) and only tie-free orders adopted; containment with the exact score as tie-breaker; NO document-frequency table -- one less structure to keep all-or-nothing |
| W-P1.3-9 | The bag cannot tell the episode that holds a short, misspelt cue IN ORDER from the dozens that hold its words: an alignment (semi-global edit distance) ranked first -- then containment-IDF, then the exact score -- took fragments 0.953 → 0.977, and collapsed keyword cues to 0.52. The keyword control is the only reason that was seen | Sellers 1980 (semi-global alignment); Damerau 1964 (transposition) | The alignment counts only when SIGNIFICANT: at most a third of the median candidate's distance (the candidates are the null), at most a third of the cue, and not contradicted by the bag (within 0.2 of the best containment). Chosen on dev; confirmed once on test and on an end-to-end protocol it never saw: fragments 0.889 → 0.922 (+182 -0 paired), keyword cues +1 -3 of 1,197 |
| W-P1.3-10 | The end-to-end lab reported "control +1 -0" beside per-arm numbers that said the control had LOST two cues | -- | A local `pl[4096]` shadowed the paired-loss counter `pl[2]`: losses were written into piece lengths. Renamed; lab programs now build with `-Wshadow`. Cross-checking two summaries of the same data is what caught it |
| W-P1.3-11 | Chunk overlap was expected to be needed for cues that straddle two episodes | Measured end to end, 26 K episodes | Overlap 64 / 128 bought 2 / 5 cues of 3,600 for 3% / 9% more text: default stays 0 |
| W-P1.3-12 | A mutant that took the median as twice the LOWER middle value survived the whole suite: no test had an even number of candidates with distinct middles at the threshold | Mutation testing | S9's two-candidate case (edits 1 and b: aligned iff b >= 5); the mutant now fails it. Caught on the final code: 8 of 8 ranking mutants, 5 of 5 alignment mutants, 5 of 5 lifecycle mutants (early eviction, id reuse, evicting the unconsolidated, stale offsets after compaction, recalling the deleted) |
| W-P1.3-13 | `engram_store_fingerprint` promised that equal fingerprints answer every query the same way, but hashed none of the configuration that decides an answer (C, rank, caps, chunking) | -- | Every such value hashed; STOREPRINT and RECALLPRINT compared across platforms by the gate |
| W-P1.3-14 | The alignment's cue bound was first reasoned as 3/2 x 480 = 720 codepoints; a folded episode can hold 4 codepoints per byte, so the true bound is 2,880 -- the store used the proven one, and S9's own "too long" cue (1,200) turned out alignable | test_align: at most ENGRAM_FOLD_OUT_MAX codepoints per byte, over random bytes | The bound is derived in code from ENGRAM_FOLD_OUT_MAX and chunk_cap; the test cue is 3,200 |
