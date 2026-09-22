/* ==================================================================================================
 * ENGRAM -- a memory that consolidates into its own weights while the machine is idle.
 * ==================================================================================================
 *
 * WHAT THIS IS, IN ONE PARAGRAPH.
 *
 * Every persistent-memory system shipping today consolidates into TEXT: it re-reads a transcript,
 * merges duplicates, prunes stale entries and writes a tidier set of notes. That is retrieval with
 * housekeeping. ENGRAM consolidates into WEIGHTS -- what was recalled often enough becomes part of
 * the model rather than part of a file it searches -- and it is able to do that for one reason:
 *
 *      EVERY WEIGHT UPDATE IS A TRANSACTION. It is measured against held-out evidence AND against a
 *      guard set of what the model already knew, and if either test fails the update is reverted
 *      BYTE-IDENTICALLY, leaving nothing behind.
 *
 * Writing to weights is what nobody does, and the reason nobody does it is that a bad write is
 * unrecoverable and a good write erodes something else. Both are removed here rather than
 * mitigated: the first by the revert, the second by DIVIDING -- material that conflicts with what a
 * shard holds is given its own shard instead of being averaged into the incumbent.
 *
 * ==================================================================================================
 * THE TWO STORES, AND WHY ONE CANNOT DO BOTH JOBS
 * ==================================================================================================
 * McClelland, McNaughton & O'Reilly (1995) is not an analogy here, it is the design. Their argument
 * is arithmetic rather than biological: a system with OVERLAPPING representations generalises well
 * and cannot absorb a single new fact without disturbing its neighbours, and a system with SPARSE
 * separated representations can absorb one exposure safely and generalises not at all. One structure
 * cannot have both properties, so a system that must do both needs two.
 *
 *      FAST   engram_epi   sparse, separated, exact, written on every interaction, never trained
 *      SLOW   the bank     distributed, generalising, written ONLY by consolidation, transactional
 *
 * The fast store is what you have said and done. The slow store is what you are like. The loop
 * between them runs while the machine is idle, which is the part that has not been built before.
 *
 * ==================================================================================================
 * WHAT IS DELIBERATELY NOT HERE
 * ==================================================================================================
 * NO PRETRAINED WEIGHTS. v1 ships with an empty bank and learns only from its user. A donor set
 * trained on somebody else's corpus would be an unproven transfer and a provenance question, and
 * neither the episodic store (a fixed projection) nor the router (k-means over keys) needs training
 * to work at all.
 *
 * NO NETWORK. Nothing in this program opens a socket. That is not a privacy feature bolted on; it
 * is the absence of the code that would be required to violate it.
 *
 * NO CLAIM THAT CONSOLIDATION HELPS, YET. The arms that would establish it -- A no consolidation,
 * Z random replay, R retrieval-ordered replay, at an identical replay budget -- are in test/ and
 * their result is whatever it is. If Z ties R, then ordering episodes by how often they were
 * recalled earned nothing, and the results file will say so.
 *
 * ==================================================================================================
 * THE RULES THIS TREE IS HELD TO
 * ==================================================================================================
 *   R1  NOTHING THAT CAN FAIL IS VOID. It returns an engram_rc. (The sticky-error buffers in
 *       engram_buf.h are the one documented shape that defers the report, and the ONLY way to take
 *       data out of them returns the rc, so the failure cannot be ignored -- only postponed.)
 *   R2  EVERY ALLOCATION GOES THROUGH engram_alloc.h, so every one of them can be failed on purpose
 *       and every failure path is exercised by a test rather than trusted.
 *   R3  EVERY FILE IS WRITTEN ATOMICALLY. A crash leaves the old complete file or the new complete
 *       file, never a torn one.
 *   R4  NOTHING IS MEASURED WITHOUT ITS CONTROL. A number that could not have come out differently
 *       is not a measurement.
 *   R5  DETERMINISM. The same inputs produce the same bytes on every run, every build, every
 *       machine -- because the projection and the hashes are recomputed rather than stored, and a
 *       single difference would read one machine's store through another machine's geometry.
 * ============================================================================================== */
#ifndef ENGRAM_H
#define ENGRAM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- VERSION ---------------------------------------------------------------------------------
 * Written into every file this program produces. A store carries the version that wrote it and the
 * loader refuses a version it does not understand, rather than reading a newer layout as if it were
 * an older one and reporting confident nonsense. MAJOR changes the on-disk meaning; MINOR adds
 * fields an older reader may safely skip. */
#define ENGRAM_VER_MAJOR 0u
#define ENGRAM_VER_MINOR 1u
#define ENGRAM_VER_PATCH 0u
#define ENGRAM_VER_STRING "0.1.0"

/* ---- THE EMBEDDING DIMENSION -----------------------------------------------------------------
 * 512 is the value the calibration sweep on the VECTRA TRACE corpus selected once that corpus was
 * made reproducible -- an earlier sweep moved to 256 and back because the directory walk followed
 * readdir order, so two copies of the same tree were two different corpora. It is a starting point
 * with provenance, not a law; P1.2's harness re-measures it on ENGRAM's own material.
 *
 * A #ifndef, because a dimension that can only be changed by editing a header is an architecture,
 * and this is a parameter. Must be a multiple of 8 (the encoder and the projection unroll by 8). */
#ifndef ENGRAM_D
#define ENGRAM_D 512u
#endif

/* ---- THE EPISODIC EXPANSION ------------------------------------------------------------------
 * EXPAND, THEN SPARSIFY, and the order is the whole mechanism.
 *
 * Taking the K largest dimensions of an embedding DIRECTLY does not work, and the failure is
 * instructive: an embedding is a structured vector whose magnitude concentrates in dimensions that
 * are large for structural reasons, so "the K largest" is very nearly a constant set and different
 * episodes receive nearly identical codes. The symptom is a cue matching PERFECTLY against the
 * WRONG episode -- similarity 1.0000 and the wrong answer, which is worse than a miss because it is
 * confident.
 *
 * So the vector is projected into a higher dimension through a fixed pseudo-random +-1 matrix
 * first -- every output is a signed sum of many inputs, so which ones win depends on the WHOLE
 * vector -- and only then sparsified.
 *
 * THE ARITHMETIC. Two random K-of-E codes share on average K*K/E dimensions:
 *
 *      K=32, E=2048    32*32/2048 = 0.50 dimensions of accidental overlap
 *      K=32, E=512     32*32/512  = 2.00              -- four times worse
 *      K=16, E=512     16*16/512  = 0.50              -- same overlap, half the signal
 *
 * E is held at four times D so the ratio is a property of the design rather than of whichever D was
 * chosen, and K keeps accidental overlap at half a dimension. */
#ifndef ENGRAM_EPI_E
#define ENGRAM_EPI_E (ENGRAM_D * 4u)
#endif
#ifndef ENGRAM_EPI_K
#define ENGRAM_EPI_K 32u
#endif

/* ---- THE FAN-IN, AND THE TRADE IT CONTROLS ----------------------------------------------------
 * How many of the D input dimensions each expanded output reads.
 *
 * At 0 an output reads ALL of them: superb separation, poor completion, because masking any input
 * perturbs every output, so a DEGRADED cue -- half a sentence, a misremembered phrase -- moves every
 * dimension a little and wins a different set.
 *
 * At F < D an output reads exactly F, so roughly (1 - m/D)^F of the outputs survive a cue masked at
 * m dimensions BIT-EXACTLY, and the degraded cue's winners are drawn largely from the same survivors
 * that won when it was stored.
 *
 * Separation and completion are the two ends of ONE dial. The default is 0 because separation is
 * what a store is for; moving it is a decision with a measurement attached, made in P1.3. */
#ifndef ENGRAM_EPI_FANIN
#define ENGRAM_EPI_FANIN 0u
#endif

/* ---- HOW MUCH TEXT AN EPISODE CARRIES ---------------------------------------------------------
 * The store holds a sparse CODE and a bounded verbatim TAIL. The code is what is matched; the tail is
 * what a human reads back. Bounded, because an unbounded tail makes the store a second copy of the
 * corpus, and the thing that must stay cheap here is the match. */
#ifndef ENGRAM_EPI_TAIL
#define ENGRAM_EPI_TAIL 480u
#endif

/* ---- COMPILE-TIME CONTRACTS ---------------------------------------------------------------------
 * These fail the BUILD, not a run. A wrong dimension is not a runtime condition to be handled; it is
 * a configuration that must never produce a binary. The negative-array trick is C99-portable. */
#define ENGRAM_STATIC_ASSERT(cond, name) typedef char engram_static_assert_##name[(cond) ? 1 : -1]

ENGRAM_STATIC_ASSERT(ENGRAM_D % 8u == 0u,                d_multiple_of_8);
ENGRAM_STATIC_ASSERT(ENGRAM_D >= 64u,                    d_at_least_64);
ENGRAM_STATIC_ASSERT(ENGRAM_EPI_E >= ENGRAM_D,           expansion_not_smaller);
ENGRAM_STATIC_ASSERT(ENGRAM_EPI_K >= 4u,                 k_at_least_4);
ENGRAM_STATIC_ASSERT(ENGRAM_EPI_K <= ENGRAM_EPI_E / 8u,  k_is_sparse);
ENGRAM_STATIC_ASSERT(ENGRAM_EPI_E <= 65535u,             e_fits_in_u16_index);
ENGRAM_STATIC_ASSERT(ENGRAM_EPI_FANIN <= ENGRAM_D,       fanin_within_d);
ENGRAM_STATIC_ASSERT(sizeof(float) == 4u,                float_is_32bit);
ENGRAM_STATIC_ASSERT(sizeof(double) == 8u,               double_is_64bit);
ENGRAM_STATIC_ASSERT(sizeof(uint64_t) == 8u,             u64_is_64bit);

/* ---- RETURN CODES ----------------------------------------------------------------------------
 * Every function that can fail returns one of these (rule R1). Negative is failure, zero is success.
 * The codes are part of the ABI and are never renumbered: a caller that logged "-4" last year must
 * still be able to look up what -4 meant. */
typedef enum {
    ENGRAM_OK         =   0,
    ENGRAM_E_ARG      =  -1,   /* a bad argument: the caller's fault, and nothing was touched      */
    ENGRAM_E_MEM      =  -2,   /* allocation failed: nothing was touched                           */
    ENGRAM_E_IO       =  -3,   /* a read, write, flush or rename failed                            */
    ENGRAM_E_FORMAT   =  -4,   /* a file or buffer is not what it claims to be                     */
    ENGRAM_E_VERSION  =  -5,   /* a file is from a version this build does not understand          */
    ENGRAM_E_EMPTY    =  -6,   /* the operation is meaningless on an empty store                   */
    ENGRAM_E_SHORT    =  -7,   /* the input is too short to carry the evidence asked of it         */
    ENGRAM_E_OVERFLOW =  -8,   /* a size computation would overflow: refused before any arithmetic */
    ENGRAM_E_AUTH     =  -9,   /* an integrity or authenticity check failed -- tampered or wrong key */
    ENGRAM_E_NOTFOUND = -10,   /* the thing asked for does not exist                               */
    ENGRAM_E_FULL     = -11,   /* a bounded structure is at its stated capacity                    */
    ENGRAM_E_STATE    = -12,   /* called in a state where the operation is not defined             */
    ENGRAM_E_UTF8     = -13,   /* text is not valid UTF-8, and is refused rather than guessed at   */
    ENGRAM_E_INTERNAL = -14    /* an invariant this code relies on did not hold -- a bug, reported */
} engram_rc;

/* A stable, human-readable name for a return code. Never NULL: an unknown code returns a string
 * saying so, because a logging path that crashes on an unexpected code destroys the one piece of
 * evidence that would have explained the unexpected code. */
const char *engram_strerror(engram_rc rc);

/* The short symbolic name ("ENGRAM_E_IO"), for logs a machine will parse. Never NULL. */
const char *engram_rcname(engram_rc rc);

/* ---- A DETERMINISTIC 64-BIT MIX ---------------------------------------------------------------
 * Used by the encoder to pick dimensions and signs, and by the episodic projection to decide its
 * +-1 matrix. It must be identical ACROSS BUILDS AND MACHINES: the projection is not stored, it is
 * recomputed, so a store written on one machine and read on another would be read through a
 * DIFFERENT projection if this ever differed. That would not crash. It would recall the wrong
 * episode, confidently, forever.
 *
 * splitmix64's finaliser (Steele, Lea & Flood 2014). Fixed constants, no platform dependence, no
 * floating point. P1.1's test pins its output against known values so a change cannot go unseen.
 *
 * It is a BIJECTION on 64-bit integers (each step -- add, xor-shift, multiply by an odd constant --
 * is invertible), so distinct inputs never share an output; the encoder's collision arguments rest
 * on that. Defined here, inline, because the encoder calls it several times per character. */
static inline uint64_t engram_mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* Combine two 64-bit values into one, order-sensitive: mix2(a,b) != mix2(b,a) in general, which is
 * what a composite key needs -- the pair (dimension 3, symbol 7) is not the pair (7, 3). */
static inline uint64_t engram_mix2(uint64_t a, uint64_t b)
{
    return engram_mix64(engram_mix64(a) ^ (b + 0x632BE59BD9B4E019ull + (a << 6) + (a >> 2)));
}

/* FNV-1a over a byte span, then finalised through engram_mix64 so the low bits are well mixed. */
uint64_t engram_hash_bytes(const void *p, size_t n, uint64_t seed);

/* ---- VERSION REPORTING ----------------------------------------------------------------------- */
typedef struct {
    unsigned    major, minor, patch;
    unsigned    d, epi_e, epi_k, epi_fanin, epi_tail;   /* the geometry this build was compiled at */
    const char *string;
    const char *platform;                                /* "linux-x86_64", "windows-x86_64", ... */
    const char *compiler;
} engram_build_info;

/* Fill `out` with what this binary IS. Stores record this so that a store written by one geometry
 * is never silently read by another. */
engram_rc engram_build_info_get(engram_build_info *out);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_H */
