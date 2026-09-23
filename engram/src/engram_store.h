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
 *   CAPACITY         hard caps on episodes and on text bytes. A full store EVICTS CONSOLIDATED
 *                    episodes -- the least recalled first, the oldest among equals, and only as many
 *                    as the new text needs; if too few are consolidated it REFUSES the add
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

/* How recall orders its candidates. FULL is the store; the other two exist so that every claim
 * about the ranking can be checked against a lesion of it (R4) -- they are what the ranking replaced.
 *   FULL   significant alignment, then containment, then the exact score (see THE RANKING below)
 *   BAG    containment, then the exact score: FULL without the alignment
 *   EXACT  the exact (Bhattacharyya) score alone: the P1.2 cascade */
typedef enum {
    ENGRAM_RANK_FULL  = 0,
    ENGRAM_RANK_BAG   = 1,
    ENGRAM_RANK_EXACT = 2
} engram_rank;

typedef struct {
    size_t         max_episodes;     /* hard cap on live episodes                     */
    size_t         max_text_bytes;   /* hard cap on live episode text                 */
    size_t         chunk_cap;        /* bytes per episode, <= ENGRAM_EPI_TAIL          */
    size_t         chunk_overlap;    /* bytes of overlap between neighbouring chunks: 0.
                                        Measured end to end (26 K episodes, cues that
                                        straddle episodes): overlap 64 / 128 bought
                                        +0.0006 / +0.0014 fragment recall -- 2 and 5
                                        cues of 3,600 -- for 3% / 9% more text. */
    unsigned       recall_c;         /* candidates re-scored exactly per query        */
    engram_rank    rank;             /* ENGRAM_RANK_FULL                              */
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

/* ---- THE RANKING ---------------------------------------------------------------------------------
 * Stage 1 (every live episode): signature containment, the C = recall_c best kept (ties: older).
 * Stage 2 (those C): three measurements of each candidate d against the cue q --
 *   contain   the fraction of q's feature mass that d holds (engram_encq_scores)
 *   exact     the Bhattacharyya similarity of the two (the P1.2 exact score)
 *   edits     the alignment distance of q to the best stretch of d (engram_align.h)
 * -- and a candidate is ALIGNED when its alignment is significant, all three of:
 *   edits <= median(edits over the C candidates) / 3     far better than the candidates at large:
 *                                                        the candidates ARE the null distribution
 *   edits <= |q| / 3                                     it explains two thirds of the cue
 *   contain >= max(contain over the candidates) - 0.2    the bag does not contradict it
 * Order: aligned before not aligned; aligned by fewer edits; then containment; then exact; then the
 * older episode. Every step is integers or IEEE + - * / sqrt (R5).
 *
 * WHY, IN NUMBERS. Recall@1, ties counted as losses, C = 50. The rule was CHOSEN on the lab's dev
 * split and then confirmed, once each, on its test split and on a second protocol it never saw:
 *                        lab: 21,811 pre-cut episodes          store, end to end: 26,139 episodes
 *                        fragments        keyword control      fragments        keyword control
 *                        dev     test     dev     test         dev     test     dev     test
 *   EXACT  (P1.2)        0.917   0.919    0.900   0.914        0.793   0.789    0.869   0.876
 *   BAG                  0.956   0.965    0.934   0.939        0.889   0.888    0.911   0.907
 *   FULL                 0.976   0.976    0.934   0.938        0.922   0.918    0.910   0.908
 *   FULL vs BAG, paired  +72-0   +40-1    +0-0    +0-1         +182-0  +163-0   +1-3    +1-0
 * (Store protocol: whole paragraphs of scale_docs.txt added through engram_store_add, cues of 12-96
 * bytes cut at any word start with 0-2 typos -- test_store S8 re-measures it through this API.)
 * Containment beats the exact score because a fragment's source is long and the exact score's norm
 * punishes length. IDF weighting was measured and earned nothing, so the store keeps no document
 * frequencies. The alignment's gain lives in SHORT cues -- 12 to 16 bytes with typos, where the bag has
 * too few pieces to go on (16 bytes, 2 typos: 0.69 -> 0.84); from 40 bytes up every ranking is at the
 * ceiling. Alignment alone, ranked first and ungated, collapsed keyword cues to 0.52: the gate is what
 * makes it safe, and the keyword control is how that is known. A held-out keyword "loss" examined by
 * hand was a cue whose three words stand together in ANOTHER episode.
 *
 * COST. At 26,139 episodes a recall takes about 11 ms, most of it stage 1, which is linear in the
 * store; making it sub-linear is the router's job (P1.4). A cue longer than 2,880 codepoints (3/2 of
 * the longest normalised episode) cannot align and skips the alignment; a shorter one costs at most
 * |q| x |d| integer steps per candidate. */

#define ENGRAM_EDITS_NONE 0xFFFFFFFFu /* no alignment computed: rank BAG/EXACT, or a cue too long to align */

typedef struct {
    uint64_t id;
    double   contain;                /* containment of the cue in the episode, [0, 1] */
    double   exact;                  /* Bhattacharyya similarity, [0, 1]              */
    uint32_t edits;                  /* alignment distance, or ENGRAM_EDITS_NONE       */
    uint32_t aligned;                /* 1 if the alignment was significant             */
} engram_hit;

/* The k best episodes for a query, best first. *n_hits <= k. ENGRAM_E_SHORT if the query has no
 * features; ENGRAM_E_FULL if the query is too long for an exact table (see engram_encq_build);
 * ENGRAM_E_MEM if the candidate arrays could not grow (nothing changed). The first hit's recall
 * counter moves -- recall is a mutation of that one counter and nothing else. */
engram_rc engram_store_recall(engram_store *s, const void *query, size_t n, engram_hit *hits,
                              size_t k, size_t *n_hits);

void      engram_store_stats_get(const engram_store *s, engram_store_stats *st);

/* A 64-bit hash of the store's LOGICAL state: every live episode's id, time, source, flags, counters
 * and text, the next id, the geometry, and every configuration value that decides an answer (caps,
 * chunking, C, rank). Two stores with equal fingerprints answer every query the same way. Capacity and
 * scratch memory are not part of it. */
uint64_t  engram_store_fingerprint(const engram_store *s);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_STORE_H */
