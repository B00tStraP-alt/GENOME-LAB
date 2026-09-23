/* ==================================================================================================
 * engram_store.h -- THE EPISODIC STORE: what ENGRAM remembers before it has learned it.
 * ==================================================================================================
 *
 * WHAT IT IS
 * ==================================================================================================
 * The fast store of a complementary-learning system: it holds EPISODES -- bounded chunks of text, cut
 * by engram_chunk.h -- verbatim, and finds them again from a half-remembered fragment. The slow store
 * (Phase 2) later consolidates them into weights; until then, this is the only place they exist.
 *
 * THE CONTRACT
 * ==================================================================================================
 *   ALL OR NOTHING   add and every other mutation either complete or leave the store's logical state
 *                    exactly as it was. Every allocation happens before anything becomes visible; the
 *                    commit itself cannot fail. test_store.c fails every allocation of an add in turn
 *                    and compares the store's fingerprint before and after.
 *   IDS              64-bit, strictly increasing, never reused, never 0, never renumbered -- not even by
 *                    compaction. An id names one episode forever; a deleted id stays deleted.
 *   CAPACITY         hard caps on episodes and on text bytes. A full store EVICTS its oldest
 *                    CONSOLIDATED episode; if no episode is consolidated it REFUSES the add
 *                    (ENGRAM_E_FULL). It never silently drops a memory that exists nowhere else.
 *   DETERMINISM      the same sequence of calls produces the same fingerprint and the same recall
 *                    results on every platform (R5). The store reads no clock: times are arguments.
 *   NOT RESIDENT     dense vectors are not stored. They are a deterministic function of the text and
 *                    cost ~15 us per episode to recompute, where storing them costs 2 KB each.
 *   THREADS          one store is used by one thread at a time; recall uses the store's own scratch.
 * ============================================================================================== */
#ifndef ENGRAM_STORE_H
#define ENGRAM_STORE_H

#include "engram.h"
#include "engram_enc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_store engram_store;

typedef struct {
    size_t         max_episodes;     /* hard cap on live episodes                     */
    size_t         max_text_bytes;   /* hard cap on live episode text                 */
    size_t         chunk_cap;        /* bytes per episode, <= ENGRAM_EPI_TAIL          */
    size_t         chunk_overlap;    /* bytes of overlap between neighbouring chunks  */
    unsigned       recall_c;         /* candidates re-scored exactly per query        */
    engram_enc_cfg enc;              /* the geometry every signature is built in      */
} engram_store_cfg;

void engram_store_cfg_default(engram_store_cfg *cfg);

/* Flags of an episode. */
#define ENGRAM_EPI_LIVE          0x01u
#define ENGRAM_EPI_CONSOLIDATED  0x02u

typedef struct {
    uint64_t    id;
    uint64_t    time_ms;             /* as given to add */
    uint32_t    source;              /* caller's tag, e.g. which file or conversation */
    uint32_t    flags;
    uint32_t    recalls;             /* times returned first by recall */
    const char *text;                /* NOT NUL-terminated; valid until the next mutation */
    size_t      len;
} engram_episode;

typedef struct {
    size_t   live, tombstones, capacity;
    size_t   text_bytes, text_capacity;
    size_t   consolidated;
    uint64_t next_id;
    uint64_t adds, deletes, evictions, refusals, compactions;
    size_t   bytes_resident;         /* everything the store holds, for the P5.5 budget */
} engram_store_stats;

engram_rc engram_store_open(engram_store **out, const engram_store_cfg *cfg);  /* cfg NULL: defaults */
void      engram_store_close(engram_store *s);

/* Cut text into episodes and add them all, or none.
 *   ENGRAM_OK       *first_id .. *first_id + *n_added - 1 are the new ids (consecutive)
 *   ENGRAM_E_SHORT  the text holds no encodable feature at all: nothing added
 *   ENGRAM_E_FULL   no room, and not enough consolidated episodes to evict: nothing added
 *   ENGRAM_E_UTF8   STRICT policy and ill-formed text: nothing added
 *   ENGRAM_E_MEM    an allocation failed: nothing added
 *   ENGRAM_E_ARG    bad arguments
 * A chunk with no encodable feature (only punctuation, say) is not stored and takes no id. */
engram_rc engram_store_add(engram_store *s, const void *text, size_t n, uint64_t time_ms,
                           uint32_t source, uint64_t *first_id, size_t *n_added);

engram_rc engram_store_delete(engram_store *s, uint64_t id);             /* E_NOTFOUND if not live   */
engram_rc engram_store_consolidated(engram_store *s, uint64_t id);       /* mark: may now be evicted */
engram_rc engram_store_get(const engram_store *s, uint64_t id, engram_episode *out);

/* Reclaim tombstones in place. Allocates nothing, so it cannot fail; ids are unchanged. */
void      engram_store_compact(engram_store *s);

typedef struct {
    uint64_t id;
    double   score;                  /* exact similarity in [0, 1] */
} engram_hit;

/* The k best episodes for a query, best first. *n_hits <= k. ENGRAM_E_SHORT if the query has no
 * features; ENGRAM_E_FULL if the query is too long for an exact table (see engram_encq_build). */
engram_rc engram_store_recall(engram_store *s, const void *query, size_t n, engram_hit *hits,
                              size_t k, size_t *n_hits);

void      engram_store_stats_get(const engram_store *s, engram_store_stats *st);

/* A 64-bit hash of the store's LOGICAL state: every live episode's id, time, source, flags, counters
 * and text, the next id, and the geometry. Two stores with equal fingerprints answer every query the
 * same way. Capacity and scratch memory are not part of it. */
uint64_t  engram_store_fingerprint(const engram_store *s);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_STORE_H */
