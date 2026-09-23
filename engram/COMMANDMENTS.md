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
| W-P1.3-15 | The plan specified EXPAND-THEN-SPARSIFY codes (D → 4D through a fixed ±1 matrix, 32 winners, a fan-in dial) as what the episodic store matches, and its own research plan said to measure it first -- and P1.3 was first committed without the measurement | Sparse distributed memory (Kanerva 1988); the fly olfactory hash (Dasgupta, Stevens & Navlakha, Science 2017) | Measured (research/p13_ranking/sparse.c, 21,811 episodes, 3,600 fragment cues): source in the top 50 for 0.28-0.35 of cues across fan-in 0/16/64/256, against 0.995 for the signature and 0.944 for the dense vector; it rescued at most 2 of the 17 the signature missed. Positive control: a chunk's own text, and its text with two typos, found it FIRST every time -- the code recognises near-copies, and a fragment is not one. NOT BUILT; the E/K/fan-in constants removed from engram.h, which now records the measurement |
| W-P1.3-16 | The plan specified LEAST-RETRIEVED eviction; the store evicted the OLDEST consolidated episode, because a slot prefix made the all-or-nothing plan trivial | -- | Least recalled first, oldest among equals, consolidated only. The plan is a recall threshold and a count, found allocation-free (a 64-bucket histogram of recall counts, a binary search only past 63) and re-applied at commit; because compaction keeps slot order and counts, it no longer has to be skipped when an add evicts. S5 proves the order, the 63+ path and a two-episode byte eviction; 4 of 4 eviction mutants caught |
| W-P1.4-1 | The router was expected to make the episodic store's linear stage 1 sub-linear. Over the dense vectors of 21,811 episodes, an IVF keeps a fragment cue's source among the signature top 50 for 0.72 of cues scanning 13% of the store and 0.955 scanning 51% (full scan: 0.995); ANN recall of the exhaustive dense top 10 at 5.8% scanned is 0.58 | IVF (Jégou, Douze & Schmid, TPAMI 2011); cluster pruning by angular bounds | The space has no bucket structure for weak cues: a fragment's nearest keys sit barely above the rest, spread over every bucket, and bucket members sit at cosine 0.3-0.5 from their centroid, so an exact angular-bound pruning (cos(φ - θ), sqrt only) prunes nothing. The store keeps its full signature scan; the router serves its specified job, expert keys, where a near-copy cue's best key is found 0.998 of the time at nprobe 8 -- and an exact search sits beside the approximate one (research/p14_router) |
| W-P1.4-2 | qsort takes no context argument, and a static one would make two routers training on two threads race | C99 §7.20.5.2 | A heapsort over key indices that takes the id array directly; training reads keys in ID order, so the centroids -- and the fingerprint -- are a function of the key set, not of the order of adds and removes (R6 proves it) |
| W-P1.4-3 | Three router-test failures were the TESTS: the fault-sweep macro reused the outer loop's counter (an infinite loop); the random-life model classified a re-add of the NEWEST id as a fresh add; and the vacuity guard fired on "add into a trained router" -- zero allocation points, a sweep that would have proved nothing | -- | A separate counter; duplicates chosen by a flag, not inferred from ids; the sweep grows its prefix until the operation must allocate and prints the prefix it used |
| W-P1.4-4 | Two mutants survived the first router suite: ties among equal scores broken toward the HIGHER id, and a train that counts itself before an allocation fails | Mutation testing | Equal-score tests (one vector under four ids, exact and approximate, with a cut at k); fault sweeps now compare every counter as well as the fingerprint. 9 of 9 router mutants caught |
| W-P1.4-5 | "Already exists" had no return code; E_ARG would have hidden a duplicate id among bad arguments | -- | ENGRAM_E_EXISTS = -15, appended (codes are never renumbered); the P1.1 walk over every code now runs from ENGRAM_RC_MIN and checks nothing lies past it |
| W-P1.5-1 | The AEAD was first written to assemble its MAC input in one allocated buffer: an allocation that could fail inside a function with no way to report it, and a copy of the whole message | RFC 8439 §2.8 -- the MAC input is a concatenation, not a buffer | Poly1305 streamed (26-bit limbs, no `__int128`, the final reduction chosen in constant time): seal and open allocate nothing and fail only on the tag, and a failed open wipes the output before it returns |
| W-P1.5-2 | Vectors typed by hand are a claim, not evidence; and C99 promises string literals only up to 4,095 characters, shorter than the HKDF and BLAKE2b vectors | C99 §5.2.4.1 | `tools/gen_crypto_vectors.py` PARSES the RFC texts (each pinned by SHA-256) and computes 437 vectors with independent implementations -- hashlib/hmac, pyca/cryptography (OpenSSL), argon2-cffi -- at every boundary length the code branches on, emitted as byte arrays. The container and the keyfile are re-implemented in Python from `engram_seal.h`'s specification and matched byte for byte, both directions |
| W-P1.5-3 | The atomic writer (P1.1) allocated the directory name AFTER the rename: an allocation failure there reported ENGRAM_E_MEM for a write that had in fact replaced the file. Found by test_persist's allocation sweep over a sealed write | ALL OR NOTHING (every allocation before anything becomes visible) | The allocation moved before step 1, so the only failure left after the commit point is the directory fsync's IO. test_core now fails every allocation of a write and requires E_MEM with the old file intact -- and was run against the old writer first, where it fails (R6) |
| W-P1.5-4 | On Windows, an allocation failure while converting a path to UTF-16 was reported as ENGRAM_E_UTF8 -- "the file name is bad" when the truth was "out of memory" -- and a directory listing that could not convert a name for lack of memory SKIPPED the file as if its name were invalid: a silently short list | -- | `engram_widen` / `engram_narrow` return a code; E_MEM travels as E_MEM; a listing that runs out of memory fails instead of omitting an entry |
| W-P1.5-5 | A wrong password and a damaged keyfile both came back as ENGRAM_E_AUTH, so a user whose keyfile was corrupted would be told the password was wrong; and a failed open could leave an earlier key in the caller's buffer | -- | From a keyfile, E_AUTH means the password is wrong and nothing else: a keyfile that fails its hash is E_FORMAT. Every keyfile function wipes the output key on entry |
| W-P1.5-6 | A plain file's SHA-256 detects corruption, not a deliberate edit -- anyone can recompute a hash -- so a sealed store could be REPLACED by a plain one of an attacker's making | Downgrade attacks (FREAK, Logjam) | A caller holding a key refuses plain files (E_AUTH). The 64-byte header is the AEAD's associated data and the kind is in the HKDF info, so no flag, length or kind can be flipped or relabelled; a fresh 32-byte salt per write gives a fresh key and nonce, so no (key, nonce) pair repeats. test_persist F3 states, as a passing test, exactly what a plain file does NOT promise |
| W-P1.5-7 | A file that stores an index beside its data can carry an index that disagrees with the data | -- | The store writes texts, not signatures; the router writes keys and centroids, not buckets. Both are recomputed on load (385 ms for the 26,139-episode store; C dot products per key for the router), so what loads is a state the API could have built, and a store written before an encoder change is re-indexed rather than refused |
| W-P1.5-8 | Is every check actually tested? 54 mutants (tools/mutants_p15.py): 21 in the primitives (round constants, rotations, padding, carries, the clamp, the AAD padding, the tag length, the wipe on failure), 12 in the container and keyfile, 11 in the store's loader and saver, 8 in the router's, 2 in the atomic writer. The first run left 6 alive. Two were TEST GAPS: an episode longer than the file's own cap was only ever refused because the doctored length also broke the parse, and no router had been searched before it was saved, so a dropped search counter went unseen | Mutation testing | Tests added that isolate each (a valid cap below an episode's length; searches before every save; every lifetime counter required non-zero before its round trip). 50 of 54 killed -- one of them (X3, W-P1.5-9) only under Wine, where the defect lives, and the runner runs it there. The 4 survivors are EQUIVALENT -- each deletes a check whose refusal a later one always makes: the store's live-count bound (pass 1 overruns first), its early configuration check (open() repeats it), the router's exact-length check (the parse then over- or under-runs) and its id-0 check (0 is the id map's empty mark, so it always reads as a duplicate). They are kept: each refuses at the point the field is read |
| W-P1.5-9 | The first P1.5 gate FAILED under Wine: "never create a keyfile over an existing one" was guarded by `engram_file_exists`, which by contract answers "no" to anything it cannot tell -- and on Windows it allocates. Under memory pressure the check said "absent" and the keyfile was REPLACED: a new salt, and every sealed file under the old key unreadable, for good. A check-then-write is also a race with any other process | TOCTOU; `link(2)` and `MoveFileEx` without `MOVEFILE_REPLACE_EXISTING` both refuse an existing name in the one step that creates it | `engram_file_create_atomic`: the same four steps as the atomic write, but the COMMIT refuses an existing target (E_EXISTS) -- no earlier check can be overtaken. The keyfile keeps a cheap pre-check, now one that reports what it cannot tell (`engram_file_size`), so a failed check is an error, never a "no". test_core fails every IO operation and allocation of a create (absent or whole, never partial); mutants X2 (the commit replaces) and X3 (the old check) are killed -- X3 only on Windows, so the runner runs it under Wine |
