/* ==================================================================================================
 * engram_alloc.h -- every allocation in ENGRAM, checked, counted, and failable on purpose.
 * ==================================================================================================
 *
 * WHY THIS EXISTS, AND IT IS NOT A WRAPPER FOR ITS OWN SAKE
 * ==================================================================================================
 * An error path that has never run is not an error path. It is a guess about what the code would do,
 * written by someone who could not watch it happen. The overwhelming majority of real allocation-
 * failure handling in C has never executed once, and the defects that live there are of the worst
 * kind: a function that fails, is declared void, and hands its caller a zeroed key or an
 * uninitialised count that looks exactly like a result.
 *
 * So every allocation in this tree goes through here (rule R2), and here it can be FAILED -- the Nth
 * one, on demand. The fault sweep (P5.3) walks N from 1 upward and fails each allocation in turn
 * across a whole operation, and requires every single failure to be a clean refusal: the right return
 * code, nothing leaked, nothing half-written, the store unchanged. That turns "we handle out of
 * memory" from a claim into a measurement.
 *
 * ==================================================================================================
 * WHAT EVERY BLOCK CARRIES
 * ==================================================================================================
 *
 *      [ header 16 B: magic | size ][ the caller's bytes ][ tail guard 8 B ]
 *                                   ^ returned pointer, 16-byte aligned
 *
 *   MAGIC     distinguishes a live block, a freed block, and a pointer that never came from here. So a
 *             DOUBLE FREE and a FOREIGN FREE are detected and counted rather than handed to the real
 *             allocator, where either would corrupt the heap silently and crash somewhere unrelated.
 *   SIZE      makes live-byte accounting exact, so a leak is a number rather than a suspicion.
 *   TAIL      a fixed pattern after the caller's region. An overrun of even one byte changes it, and
 *             the damage is caught at free time on EVERY build -- including Windows release builds,
 *             where AddressSanitizer is not available.
 *
 * A block found corrupt is COUNTED and deliberately LEAKED rather than freed: handing a block with a
 * damaged header to free() would turn a detected error into an undetected one. The test suite asserts
 * the corruption counter is zero after every test, so a leak of this kind never goes unnoticed.
 *
 * ==================================================================================================
 * THE QUARANTINE, AND THE DEFECT IT EXISTS TO CLOSE
 * ==================================================================================================
 * A magic number alone does NOT detect a double free, and the reason is specific. The instant a block
 * is handed back to glibc, its tcache writes a freelist pointer and a key into the first sixteen bytes
 * of that block -- exactly where the header lives. The DEAD marker written a moment earlier is gone
 * before anyone can read it. Worse: once the allocator REUSES that memory for a new engram block, the
 * header reads LIVE again, and a stale second free of the OLD pointer frees the NEW allocation. That
 * is a use-after-free in the caller, and no check on the header can see it.
 *
 * So freed blocks are not returned at once. They are POISONED (every usable byte set to 0xDD, so a
 * read-after-free returns obvious garbage rather than plausible stale data), marked DEAD, and held in
 * a FIFO ring. While a block sits there nobody else can be given its memory, so its DEAD marker
 * survives and a second free of it is identified exactly. When the ring is full the oldest block is
 * released for real.
 *
 * This is the mechanism AddressSanitizer uses, and it has the same honest limit: a double free that
 * arrives AFTER the block has left quarantine and been reused cannot be told apart from a legitimate
 * free of the new block. The quarantine makes that window long rather than zero. A block larger than
 * the quarantine's byte budget is released immediately, so the ring can never pin a large amount of
 * memory -- and it is released AFTER poisoning, so a read-after-free still sees 0xDD until reuse.
 * ============================================================================================== */
#ifndef ENGRAM_ALLOC_H
#define ENGRAM_ALLOC_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- ALLOCATION ------------------------------------------------------------------------------
 * Semantics are malloc's, with three deliberate differences, each of which removes an ambiguity:
 *
 *   engram_malloc(0)       returns a valid unique pointer to zero usable bytes, never NULL, so NULL
 *                          ALWAYS means failure.
 *   engram_realloc(p, 0)   is refused: returns NULL and leaves p valid and untouched. realloc-to-zero
 *                          is implementation-defined in C and callers disagree about what NULL then
 *                          means; here there is nothing to disagree about.
 *   engram_free(NULL)      is a no-op, as for free.                                                */
void *engram_malloc(size_t n);
void *engram_calloc(size_t count, size_t size);            /* zeroed; overflow-checked            */
void *engram_realloc(void *p, size_t n);
void  engram_free(void *p);

/* count * size bytes, refusing (NULL) if the product would overflow size_t -- checked BEFORE the
 * multiplication, since after it the evidence is gone. Not zeroed; use engram_calloc for that. */
void *engram_array(size_t count, size_t size);

/* Grow an array to hold at least `need` elements of `size` bytes. *cap is doubled from max(*cap, 8)
 * until it suffices. On failure *pp and *cap are UNCHANGED, so the caller's array is still valid and
 * nothing was lost -- the property the realloc idiom `p = realloc(p, n)` famously lacks. */
engram_rc engram_grow(void **pp, size_t *cap, size_t need, size_t size);

char *engram_strdup(const char *s);
char *engram_strndup(const char *s, size_t n);

/* ---- INTROSPECTION --------------------------------------------------------------------------- */

/* The usable size of a live block, or 0 for NULL or a pointer that fails verification. */
size_t engram_alloc_size(const void *p);

/* Verify one block's header and tail without freeing it. ENGRAM_OK, or ENGRAM_E_INTERNAL if damaged. */
engram_rc engram_alloc_check(const void *p);

typedef struct {
    uint64_t n_alloc;          /* successful fresh allocations (malloc/calloc/array/strdup)     */
    uint64_t n_realloc;        /* successful reallocs                                           */
    uint64_t n_free;           /* successful frees                                               */
    uint64_t n_fail_injected;  /* failures caused on purpose by engram_alloc_fail_at             */
    uint64_t n_fail_real;      /* failures the system allocator actually returned                */
    uint64_t n_double_free;    /* frees of a block already freed -- must be 0                    */
    uint64_t n_foreign_free;   /* frees of a pointer that never came from here -- must be 0      */
    uint64_t n_overrun;        /* blocks whose tail guard was damaged -- must be 0               */
    uint64_t live_blocks;      /* allocated and not yet freed                                    */
    uint64_t live_bytes;       /* usable bytes in live blocks                                    */
    uint64_t peak_bytes;       /* the high-water mark of live_bytes                              */
    uint64_t quarantine_blocks;/* freed, poisoned, held back from reuse                          */
    uint64_t quarantine_bytes; /* usable bytes they held                                         */
} engram_alloc_stats;

void engram_alloc_stats_get(engram_alloc_stats *out);

/* The sum of the three "must be 0" counters. The single number a test asserts on. */
uint64_t engram_alloc_corruption(void);

/* ---- FAULT INJECTION -------------------------------------------------------------------------
 * Fail the Nth allocation from now (1-based), then behave normally. 0 disables.
 * Counts every engram_malloc/calloc/realloc/array/grow/strdup call that reaches the allocator. */
void     engram_alloc_fail_at(uint64_t nth);
uint64_t engram_alloc_calls(void);        /* allocation attempts since the last reset */

/* Reset the counters that describe a RUN (calls, injected failures, the fault target, the peak).
 * Live-block and corruption counters are NOT reset: they describe the heap, which a reset does not
 * change, and resetting them would let a leak from one test hide inside the next. */
void engram_alloc_reset_run(void);

/* Release every quarantined block for real. Call before a leak check or at exit, so that memory the
 * quarantine is legitimately holding is not reported by an external leak detector as lost. */
void engram_alloc_quarantine_flush(void);

/* ---- CONFIGURATION ---------------------------------------------------------------------------
 * The quarantine ring's length and byte budget. 0 blocks disables it (every free releases at once),
 * which a throughput build may want; the test build never does. */
#ifndef ENGRAM_ALLOC_QUARANTINE_N
#define ENGRAM_ALLOC_QUARANTINE_N 256u
#endif
#ifndef ENGRAM_ALLOC_QUARANTINE_BYTES
#define ENGRAM_ALLOC_QUARANTINE_BYTES ((size_t)64u << 20)     /* 64 MiB */
#endif

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_ALLOC_H */
