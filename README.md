# ENGRAM

A private memory for a person's computer that **learns into its own weights while the machine is idle**,
and can prove, for every single thing it learned, that learning it broke nothing it already knew.

Written in C99 with no dependencies, to ship as one static Windows EXE (Phase 6); CPU only; the same bits
on every machine (the gate compares the suites' fingerprints, bit for bit, across Linux and Windows).

Everything lives in [`engram/`](engram/). How the work is done: [`engram/COMMANDMENTS.md`](engram/COMMANDMENTS.md)
(the rules, and the ledger of every wall met). What is built, in what order, and what must be proven:
[`engram/ROADMAP.md`](engram/ROADMAP.md).

## Where it stands

| | |
|---|---|
| **Phase 1** — the ground | Done and gated: P1.1–P1.5 |
| **Phase 2** — the slow store: weights that learn | P2.1 (the expert) done and gated; P2.2.1 (the backward pass) done; **next: P2.2.2** |
| Phases 3–6 | Not started |

Phase 1 had 5 sub-phases. From Phase 2 on, each phase has 5 sub-phases of 3 mini-phases (.1 research or
design, .2 build, .3 proof and the sub-phase's gate): 75 mini-phases. None of the breakthrough claims
B1–B8 in the roadmap is claimed yet.

## Layout

```
engram/
  src/        the library -- one module a file pair, all of it C99
  test/       one suite per module group; generated reference vectors (*_vectors.h); test/data: corpora
  tools/      the gate, the vector generators and the Python references they use, mutation campaigns
  research/   the measurements behind each decision, with their results (p13_ranking, p14_router,
              cascade_weight, p2_slow)
  Makefile
```

| Module | What it is |
|---|---|
| `engram.h` | the version, the return codes, the shared constants |
| `engram_alloc` | every allocation checked, counted, and failable on purpose |
| `engram_plat` | the operating system: files (atomic write and create), time, paths, Linux and Windows |
| `engram_log`, `engram_rng`, `engram_buf` | the log; reproducible randomness; bounded little-endian IO |
| `engram_text`, `engram_unicode_tables` | UTF-8, character classes, folding (Unicode 14.0.0) |
| `engram_enc`, `engram_chunk`, `engram_align` | text to vector, signature and exact score; episodes; the alignment |
| `engram_store` | the episodic store: what ENGRAM remembers before it has learned it |
| `engram_router` | an IVF index over unit keys: which keys lie near a query |
| `engram_crypto`, `engram_seal` | SHA-256, HMAC, HKDF, BLAKE2b, Argon2id, ChaCha20-Poly1305; the sealed container and the keyfile |
| `engram_math` | exp and log2 from + − × ÷ alone: the same bits everywhere |
| `engram_cascade` | the Cascade weight: a ternary weight whose position is its whole training state |
| `engram_ctx` | the hashed contexts an expert reads |
| `engram_expert` | one expert of the slow store: forward pass, bits, backward pass |

## Build and test

Needs `gcc`, `make`; for Windows `x86_64-w64-mingw32-gcc` and `wine`; for the generators `python3` with
`numpy`.

```
cd engram
make test         # Linux: every suite
make win          # Windows: static PE32+ test programs
make wine-test    # the Windows programs under Wine
make asan         # AddressSanitizer + UndefinedBehaviorSanitizer
make tsan         # ThreadSanitizer
make gate         # the full gate: all of the above, cross-platform identity, and proof the gate can fail
```

Every variant builds with `-Werror` and `-ffp-contract=off` (see the Makefile for why the second is not
optional).
