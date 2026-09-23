/* ==================================================================================================
 * engram_router.h -- THE ROUTER: which of many keys lie near a query, without scoring all of them.
 * ==================================================================================================
 *
 * WHAT IT IS
 * ==================================================================================================
 * An inverted-file (IVF) index over unit KEYS (Jegou, Douze & Schmid, TPAMI 2011, after Sivic &
 * Zisserman's "Video Google", ICCV 2003). A coarse quantiser -- C centroids fitted by spherical
 * k-means -- splits the keys into C buckets, each key in the bucket of its nearest centroid. A query
 * scores the C centroids, then only the keys in the NPROBE nearest buckets:
 *
 *      cost per query  =  (C + N * nprobe / C) dot products   minimised near C = sqrt(N * nprobe)
 *
 * The keys it is built for are the slow store's experts (P2.4: "is this expert relevant?" must cost
 * dot products, not a forward pass) -- and anything else that is a unit vector with an id.
 *
 * MEASURED (test_router R7/R8; 26,139 keys = the dense vectors of every scale-corpus episode;
 * C = 161 = sqrt(N); nprobe 8, scoring 6-9% of the keys):
 *   near-copy cue (an episode with two typos)   its exhaustive best key found 0.998 of the time
 *   fragment cue (a fifth of an episode)        best key 0.69, exhaustive top 10 0.63
 * The second line is the space, not the index: a fragment's nearest keys sit at cosines barely above
 * the rest and are spread over every bucket (research/p14_router). That is why the EPISODIC STORE does
 * not route through this index -- to keep the source among its candidates for 95% of fragment cues it
 * would have to scan half the store (W-P1.4-1) -- and why a caller that needs the true best key, and
 * can afford C dot products plus a scan, asks engram_router_exact. Trained on half the keys with the
 * other half added afterwards: the same recall. An insert costs C dot products and an append (~60-80 us
 * at C = 161).
 *
 * THE CONTRACT
 * ==================================================================================================
 *   GROWS WITHOUT A REBUILD   a key added after training goes to its nearest centroid's bucket: one
 *                             pass over the centroids and an append. An index that must be rebuilt
 *                             to accept knowledge is a snapshot, not a memory. Training again is a
 *                             choice the caller makes when the buckets have drifted out of balance
 *                             (the stats say how far), never a precondition of adding.
 *   COUNTED, NOT ESTIMATED    every search reports exactly how many keys and centroids it scored.
 *   EXACT ON DEMAND           engram_router_exact scores every key: the reference the approximate
 *                             search is measured against. Before training, and whenever nprobe
 *                             covers every bucket, the search IS exact -- the same hits, the same bits.
 *   ALL OR NOTHING            add and train allocate everything first; a failure changes nothing.
 *                             remove cannot fail. The fingerprint proves both under fault injection.
 *   DETERMINISTIC             k-means is seeded, every sum runs in a fixed order, ties go to the lower
 *                             index or id: the same calls give the same centroids and the same hits
 *                             on every platform (R5).
 *   THREADS                   one router is used by one thread at a time; search uses its scratch.
 * ============================================================================================== */
#ifndef ENGRAM_ROUTER_H
#define ENGRAM_ROUTER_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct engram_router engram_router;

#define ENGRAM_ROUTER_SEED 0x524F555445ull     /* "ROUTE"                                       */
#define ENGRAM_ROUTER_UNIT_TOL 1e-3            /* | |key| - 1 | allowed: keys and queries are unit */

typedef struct {
    unsigned dim;            /* key dimension, 8..4096 and a multiple of 8; default ENGRAM_D      */
    unsigned nprobe;         /* buckets a search scores when the caller passes 0; default 8      */
    unsigned iters;          /* Lloyd iterations at most; default 16                             */
    size_t   train_max;      /* keys k-means sees (a deterministic sample beyond it); 0 = all;
                                default 65536                                                    */
    uint64_t seed;           /* k-means++ seeding; default ENGRAM_ROUTER_SEED                     */
} engram_router_cfg;

void engram_router_cfg_default(engram_router_cfg *cfg);

engram_rc engram_router_open(engram_router **out, const engram_router_cfg *cfg);   /* NULL: defaults */
void      engram_router_close(engram_router *r);

/* Add a key under an id (any value but 0).
 *   ENGRAM_E_ARG     NULL key, id 0, a non-finite component, or a norm off 1 by more than the tolerance
 *   ENGRAM_E_EXISTS  the id is already present
 *   ENGRAM_E_MEM     nothing changed */
engram_rc engram_router_add(engram_router *r, uint64_t id, const float *key);

/* Remove a key. ENGRAM_E_NOTFOUND if absent. Allocates nothing; cannot otherwise fail. */
engram_rc engram_router_remove(engram_router *r, uint64_t id);

/* Copy a key out (dim floats). ENGRAM_E_NOTFOUND if absent. */
engram_rc engram_router_get(const engram_router *r, uint64_t id, float *key);

/* Fit C centroids to the current keys (spherical k-means, k-means++ seeding) and put every key in
 * its nearest centroid's bucket. 1 <= C <= keys. ENGRAM_E_EMPTY with no keys; ENGRAM_E_MEM leaves the
 * previous index exactly as it was. */
engram_rc engram_router_train(engram_router *r, unsigned C);

typedef struct {
    uint64_t id;
    float    score;          /* cosine of query and key */
} engram_route_hit;

typedef struct {
    size_t   keys_scored;    /* keys whose dot product this search computed  */
    size_t   centroids_scored;
    unsigned buckets_probed;
} engram_route_cost;

/* The k keys of highest cosine among the nprobe nearest buckets (nprobe 0: the configured default),
 * best first, ties to the lower id. *n_hits <= k. cost may be NULL.
 *   ENGRAM_E_ARG   bad arguments, or a query that is not a finite unit vector
 *   ENGRAM_E_MEM   the scratch could not grow; nothing changed */
engram_rc engram_router_search(engram_router *r, const float *q, size_t k, unsigned nprobe,
                               engram_route_hit *hits, size_t *n_hits, engram_route_cost *cost);

/* The same over EVERY key: the reference the search is measured against. */
engram_rc engram_router_exact(engram_router *r, const float *q, size_t k, engram_route_hit *hits,
                              size_t *n_hits, engram_route_cost *cost);

typedef struct {
    size_t   keys, buckets, empty_buckets, largest_bucket;
    double   imbalance;      /* sum of squared bucket sizes / (keys^2 / buckets): 1 when perfectly
                                even; the expected search cost grows with it                      */
    int      trained;
    uint64_t adds, removes, trains, searches;
    size_t   bytes_resident;
} engram_router_stats;

void      engram_router_stats_get(const engram_router *r, engram_router_stats *st);

/* A 64-bit hash of the LOGICAL state: configuration, centroids, and every (id, key, bucket) -- as a
 * set, so two routers holding the same keys in the same buckets agree however they got there. */
uint64_t  engram_router_fingerprint(const engram_router *r);

/* Verify every internal invariant (each key in exactly one bucket at the recorded position, the id
 * map exact, buckets consistent with the counts). ENGRAM_OK, or ENGRAM_E_INTERNAL. For tests; cost
 * O(keys + buckets). */
engram_rc engram_router_check(const engram_router *r);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_ROUTER_H */
