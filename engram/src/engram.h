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

/* ---- EXPAND-THEN-SPARSIFY: MEASURED, AND NOT BUILT (P1.3, ledger W-P1.3-15) ---------------------
 * The plan called for the episodic store to match SPARSE CODES: the dense vector projected through a
 * fixed pseudo-random +-1 matrix to E = 4D outputs, the K = 32 largest kept (accidental overlap
 * K*K/E = half a dimension), and a FAN-IN dial trading separation (each output reads all D inputs)
 * against completion (each reads F). It was built in the lab and measured before any of it entered
 * the store (research/p13_ranking/sparse.c, 21,811 episodes, 3,600 fragment cues):
 *
 *      candidate recall of the source      top 10   top 50   top 200
 *      signature (what the store uses)     0.990    0.995    0.998
 *      dense vector, D = 512               0.888    0.944    0.971
 *      sparse code, fan-in 0 / 16 / 64     0.18-0.22  0.28-0.35  0.45-0.53
 *
 * and as a positive control, a chunk's OWN text -- and its text with two typos -- found it first
 * every time at every fan-in. The code does exactly what it was designed to do: it recognises a
 * near-copy. A fragment is not a near-copy: its dense vector keeps a fraction of the source's mass,
 * and the top-K of the expansion amplifies that difference into different winners. The fan-in dial
 * moves 0.28 to 0.35. It rescued at most 2 of the 17 cues the signature's top 50 missed. Nothing in
 * the store matches sparse codes, so there are no E, K or fan-in constants to configure. */

/* ---- HOW MUCH TEXT AN EPISODE CARRIES ---------------------------------------------------------
 * An episode is at most this many bytes of verbatim text (engram_chunk.h cuts longer text), and it is
 * matched through its 1 KB signature and, for the few candidates that survive, its text. Bounded,
 * because a signature saturates on long text (P1.2) and the exact and alignment stages cost in
 * proportion to the episode. */
#ifndef ENGRAM_EPI_TAIL
#define ENGRAM_EPI_TAIL 480u
#endif

/* ---- COMPILE-TIME CONTRACTS ---------------------------------------------------------------------
 * These fail the BUILD, not a run. A wrong dimension is not a runtime condition to be handled; it is
 * a configuration that must never produce a binary. The negative-array trick is C99-portable. */
#define ENGRAM_STATIC_ASSERT(cond, name) typedef char engram_static_assert_##name[(cond) ? 1 : -1]

ENGRAM_STATIC_ASSERT(ENGRAM_D % 8u == 0u,                d_multiple_of_8);
ENGRAM_STATIC_ASSERT(ENGRAM_D >= 64u,                    d_at_least_64);
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
    ENGRAM_E_INTERNAL = -14,   /* an invariant this code relies on did not hold -- a bug, reported */
    ENGRAM_E_EXISTS   = -15    /* the thing to be created already exists: nothing was touched      */
} engram_rc;

/* The most negative code: every value from here to ENGRAM_OK is a defined code. */
#define ENGRAM_RC_MIN ENGRAM_E_EXISTS

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
    unsigned    d, epi_tail;                            /* the geometry this build was compiled at */
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
