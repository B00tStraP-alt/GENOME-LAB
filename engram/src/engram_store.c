/* ==================================================================================================
 * engram_store.c -- the episodic store. Contract in engram_store.h.
 * ==================================================================================================
 * LAYOUT. Struct of arrays over SLOTS: a slot is an episode ever added and not yet compacted away;
 * live episodes and tombstones both occupy one. Slot order is id order, always -- ids are handed out
 * increasing, slots are appended, and compaction moves slots down without reordering -- so an id is
 * found by binary search and "oldest" means "lowest slot".
 *
 * THE ADD, IN FOUR STEPS, AND WHY THE COMMIT CANNOT FAIL
 *   1. MEASURE  cut the text and sign every chunk into a stack buffer: how many chunks are storable
 *               and how many bytes they carry. Nothing allocated, nothing changed.
 *   2. PLAN     how many consolidated episodes must go to make room, as a prefix of slots [0, stop).
 *               Not enough: ENGRAM_E_FULL, nothing changed.
 *   3. RESERVE  compact if that alone makes room (allocation-free, logically invisible), else grow
 *               every array. A failed growth leaves each array's CONTENT as it was.
 *   4. WRITE, THEN COMMIT  new episodes are written into slots at and beyond n -- invisible, because
 *               everything reads only [0, n). Then the planned evictions are applied and n, the byte
 *               counts and next_id move. No step of 4 can fail.
 * ============================================================================================== */
#include "engram_store.h"
#include "engram_alloc.h"
#include "engram_chunk.h"

#include <string.h>

typedef struct { double s; size_t slot; } engram_cand;     /* a recall candidate */

struct engram_store {
    engram_store_cfg cfg;
    uint64_t         geometry;
    size_t           n;                    /* slots in use: live + tombstones           */
    size_t           cap_id, cap_time, cap_src, cap_flags, cap_rec, cap_off, cap_len, cap_sig;
    uint64_t        *id, *time_ms;
    uint32_t        *source, *flags, *recalls;
    size_t          *off;
    uint16_t        *len;
    uint64_t        *sig;                  /* n * ENGRAM_SIG_WORDS                      */
    char            *text;
    size_t           text_used, text_cap;
    size_t           live, live_bytes, consolidated;
    uint64_t         next_id;
    uint64_t         adds, deletes, evictions, refusals, compactions;
    engram_encq     *q;                    /* recall scratch                            */
    engram_cand     *cand;
    size_t           cap_cand;
};

void engram_store_cfg_default(engram_store_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->max_episodes   = 100000u;
    cfg->max_text_bytes = (size_t)64u << 20;
    cfg->chunk_cap      = ENGRAM_EPI_TAIL;
    cfg->chunk_overlap  = 0u;
    cfg->recall_c       = 50u;
    engram_enc_cfg_default(&cfg->enc);
}

static engram_rc engram_store_cfg_check(const engram_store_cfg *c)
{
    engram_chunkit it;
    if (!c || !c->max_episodes || !c->max_text_bytes || !c->recall_c) return ENGRAM_E_ARG;
    if (engram_enc_cfg_check(&c->enc) != ENGRAM_OK || c->enc.tf != ENGRAM_TF_SQRT) return ENGRAM_E_ARG;
    return engram_chunkit_init(&it, "", 0u, c->chunk_cap, c->chunk_overlap);
}

engram_rc engram_store_open(engram_store **out, const engram_store_cfg *cfg)
{
    engram_store *s;
    engram_store_cfg d;
    engram_rc rc;
    if (!out) return ENGRAM_E_ARG;
    *out = NULL;
    if (!cfg) { engram_store_cfg_default(&d); cfg = &d; }
    rc = engram_store_cfg_check(cfg);
    if (rc != ENGRAM_OK) return rc;
    s = (engram_store *)engram_malloc(sizeof *s);
    if (!s) return ENGRAM_E_MEM;
    memset(s, 0, sizeof *s);
    s->cfg = *cfg;
    s->geometry = engram_enc_geometry(&cfg->enc, ENGRAM_D);
    s->next_id = 1u;
    s->q = (engram_encq *)engram_malloc(sizeof *s->q);
    if (!s->q) { engram_free(s); return ENGRAM_E_MEM; }
    *out = s;
    return ENGRAM_OK;
}

void engram_store_close(engram_store *s)
{
    if (!s) return;
    engram_free(s->id); engram_free(s->time_ms); engram_free(s->source); engram_free(s->flags);
    engram_free(s->recalls); engram_free(s->off); engram_free(s->len); engram_free(s->sig);
    engram_free(s->text); engram_free(s->q); engram_free(s->cand);
    engram_free(s);
}

/* ---- lookup ---------------------------------------------------------------------------------- */
static int engram_store_slot(const engram_store *s, uint64_t id, size_t *slot)
{
    size_t lo = 0, hi = s->n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2u;
        if (s->id[mid] < id) lo = mid + 1u; else hi = mid;
    }
    if (lo < s->n && s->id[lo] == id && (s->flags[lo] & ENGRAM_EPI_LIVE)) { *slot = lo; return 1; }
    return 0;
}

/* ---- compaction: slots move down in order, text moves down in order; nothing is allocated ---- */
void engram_store_compact(engram_store *s)
{
    size_t r, w = 0, tw = 0;
    if (!s || s->n == s->live) return;
    for (r = 0; r < s->n; r++) {
        if (!(s->flags[r] & ENGRAM_EPI_LIVE)) continue;
        if (s->off[r] != tw) memmove(s->text + tw, s->text + s->off[r], s->len[r]);
        if (w != r) {
            s->id[w] = s->id[r]; s->time_ms[w] = s->time_ms[r]; s->source[w] = s->source[r];
            s->flags[w] = s->flags[r]; s->recalls[w] = s->recalls[r]; s->len[w] = s->len[r];
            memcpy(s->sig + w * ENGRAM_SIG_WORDS, s->sig + r * ENGRAM_SIG_WORDS,
                   ENGRAM_SIG_WORDS * sizeof *s->sig);
        }
        s->off[w] = tw;
        tw += s->len[r];
        w++;
    }
    s->n = w;
    s->text_used = tw;
    s->compactions++;
}

/* ---- add ------------------------------------------------------------------------------------- */
static engram_rc engram_store_reserve(engram_store *s, size_t slots, size_t bytes)
{
    size_t need = s->n + slots;
    if (engram_grow((void **)&s->id, &s->cap_id, need, sizeof *s->id) != ENGRAM_OK ||
        engram_grow((void **)&s->time_ms, &s->cap_time, need, sizeof *s->time_ms) != ENGRAM_OK ||
        engram_grow((void **)&s->source, &s->cap_src, need, sizeof *s->source) != ENGRAM_OK ||
        engram_grow((void **)&s->flags, &s->cap_flags, need, sizeof *s->flags) != ENGRAM_OK ||
        engram_grow((void **)&s->recalls, &s->cap_rec, need, sizeof *s->recalls) != ENGRAM_OK ||
        engram_grow((void **)&s->off, &s->cap_off, need, sizeof *s->off) != ENGRAM_OK ||
        engram_grow((void **)&s->len, &s->cap_len, need, sizeof *s->len) != ENGRAM_OK ||
        engram_grow((void **)&s->sig, &s->cap_sig, need * ENGRAM_SIG_WORDS, sizeof *s->sig) != ENGRAM_OK ||
        engram_grow((void **)&s->text, &s->text_cap, s->text_used + bytes + 1u, 1u) != ENGRAM_OK)
        return ENGRAM_E_MEM;
    return ENGRAM_OK;
}

engram_rc engram_store_add(engram_store *s, const void *text, size_t n, uint64_t time_ms,
                           uint32_t source, uint64_t *first_id, size_t *n_added)
{
    uint64_t tmp[ENGRAM_SIG_WORDS];
    engram_chunkit it;
    size_t off, len, k = 0, bytes = 0, stop = 0, freed_slots = 0, freed_bytes = 0, slot;
    engram_rc rc, last = ENGRAM_E_SHORT;

    if (first_id) *first_id = 0;
    if (n_added) *n_added = 0;
    if (!s || !first_id || !n_added || (!text && n)) return ENGRAM_E_ARG;

    /* 1. MEASURE */
    rc = engram_chunkit_init(&it, text, n, s->cfg.chunk_cap, s->cfg.chunk_overlap);
    if (rc != ENGRAM_OK) return rc;
    while (engram_chunkit_next(&it, &off, &len)) {
        rc = engram_encode_sig(&s->cfg.enc, (const char *)text + off, len, tmp, NULL);
        if (rc == ENGRAM_OK) { k++; bytes += len; }
        else if (rc == ENGRAM_E_UTF8) return rc;
        else if (rc != ENGRAM_E_SHORT) return rc;
    }
    if (k == 0) return last;

    /* 2. PLAN: evict the oldest consolidated live episodes, as a prefix of slots [0, stop) */
    if (k > s->cfg.max_episodes || bytes > s->cfg.max_text_bytes) { s->refusals++; return ENGRAM_E_FULL; }
    {
        size_t need_slots = s->live + k > s->cfg.max_episodes ? s->live + k - s->cfg.max_episodes : 0u;
        size_t need_bytes = s->live_bytes + bytes > s->cfg.max_text_bytes
                                ? s->live_bytes + bytes - s->cfg.max_text_bytes : 0u;
        for (stop = 0; stop < s->n && (freed_slots < need_slots || freed_bytes < need_bytes); stop++)
            if ((s->flags[stop] & (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) ==
                (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) {
                freed_slots++;
                freed_bytes += s->len[stop];
            }
        if (freed_slots < need_slots || freed_bytes < need_bytes) { s->refusals++; return ENGRAM_E_FULL; }
    }

    /* 3. RESERVE. Compaction is logically invisible, so it may run before a later failure. It would
     * move slots, and the eviction plan is a slot prefix -- so compact only when nothing is to be
     * evicted in this add. */
    if (stop == 0 && s->n > s->live &&
        (s->n + k > s->cap_id || s->text_used + bytes + 1u > s->text_cap))
        engram_store_compact(s);
    rc = engram_store_reserve(s, k, bytes);
    if (rc != ENGRAM_OK) return rc;

    /* 4. WRITE into slots n .. n+k-1 (invisible), THEN COMMIT */
    {
        size_t w = s->n, tw = s->text_used;
        uint64_t nid = s->next_id;
        engram_chunkit it2;
        (void)engram_chunkit_init(&it2, text, n, s->cfg.chunk_cap, s->cfg.chunk_overlap);
        while (engram_chunkit_next(&it2, &off, &len)) {
            if (engram_encode_sig(&s->cfg.enc, (const char *)text + off, len,
                                  s->sig + w * ENGRAM_SIG_WORDS, NULL) != ENGRAM_OK) continue;
            memcpy(s->text + tw, (const char *)text + off, len);
            s->id[w] = nid++;
            s->time_ms[w] = time_ms;
            s->source[w] = source;
            s->flags[w] = ENGRAM_EPI_LIVE;
            s->recalls[w] = 0;
            s->off[w] = tw;
            s->len[w] = (uint16_t)len;
            tw += len;
            w++;
        }
        /* commit */
        for (slot = 0; slot < stop; slot++)
            if ((s->flags[slot] & (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) ==
                (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) {
                s->flags[slot] &= ~(uint32_t)(ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED);
                memset(s->sig + slot * ENGRAM_SIG_WORDS, 0, ENGRAM_SIG_WORDS * sizeof *s->sig);
                s->live--;
                s->live_bytes -= s->len[slot];
                s->consolidated--;
                s->evictions++;
            }
        *first_id = s->next_id;
        *n_added = k;
        s->n = w;
        s->text_used = tw;
        s->live += k;
        s->live_bytes += bytes;
        s->next_id = nid;
        s->adds++;
    }
    /* A store that evicts on every add never takes the pre-add compaction, so tombstones and their
     * text would accumulate without bound. After the commit, compaction is still allocation-free and
     * logically invisible, so the bound is enforced here: tombstones never exceed a quarter of the
     * live episodes plus a small constant. */
    if (s->n - s->live > s->live / 4u + 64u) engram_store_compact(s);
    return ENGRAM_OK;
}

/* ---- the other mutations --------------------------------------------------------------------- */
engram_rc engram_store_delete(engram_store *s, uint64_t id)
{
    size_t slot;
    if (!s) return ENGRAM_E_ARG;
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    if (s->flags[slot] & ENGRAM_EPI_CONSOLIDATED) s->consolidated--;
    s->flags[slot] &= ~(uint32_t)(ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED);
    memset(s->sig + slot * ENGRAM_SIG_WORDS, 0, ENGRAM_SIG_WORDS * sizeof *s->sig);
    s->live--;
    s->live_bytes -= s->len[slot];
    s->deletes++;
    return ENGRAM_OK;
}

engram_rc engram_store_consolidated(engram_store *s, uint64_t id)
{
    size_t slot;
    if (!s) return ENGRAM_E_ARG;
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    if (!(s->flags[slot] & ENGRAM_EPI_CONSOLIDATED)) { s->flags[slot] |= ENGRAM_EPI_CONSOLIDATED; s->consolidated++; }
    return ENGRAM_OK;
}

engram_rc engram_store_get(const engram_store *s, uint64_t id, engram_episode *out)
{
    size_t slot;
    if (!s || !out) return ENGRAM_E_ARG;
    memset(out, 0, sizeof *out);
    if (!engram_store_slot(s, id, &slot)) return ENGRAM_E_NOTFOUND;
    out->id = s->id[slot]; out->time_ms = s->time_ms[slot]; out->source = s->source[slot];
    out->flags = s->flags[slot]; out->recalls = s->recalls[slot];
    out->text = s->text + s->off[slot]; out->len = s->len[slot];
    return ENGRAM_OK;
}

/* ---- recall ----------------------------------------------------------------------------------
 * Stage 1: every live episode's signature containment. The C best, by a size-C min-heap whose
 * order is (score, then LOWER slot wins) -- a total order, so the candidate set is deterministic.
 * Stage 2: the exact score of each candidate; the k best returned, ties to the lower id. */
static int engram_cand_worse(const engram_cand *a, const engram_cand *b)
{
    return a->s < b->s || (a->s == b->s && a->slot > b->slot);
}

static void engram_heap_sift(engram_cand *h, size_t n, size_t i)
{
    for (;;) {
        size_t l = 2u * i + 1u, r = l + 1u, m = i;
        engram_cand t;
        if (l < n && engram_cand_worse(&h[l], &h[m])) m = l;
        if (r < n && engram_cand_worse(&h[r], &h[m])) m = r;
        if (m == i) return;
        t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

engram_rc engram_store_recall(engram_store *s, const void *query, size_t n, engram_hit *hits,
                              size_t k, size_t *n_hits)
{
    engram_cand *heap;
    size_t C, hn = 0, slot, i, j;
    engram_rc rc;
    if (n_hits) *n_hits = 0;
    if (!s || !hits || !n_hits || !k || (!query && n)) return ENGRAM_E_ARG;
    rc = engram_encq_build(s->q, &s->cfg.enc, query, n);
    if (rc != ENGRAM_OK) return rc;
    if (s->live == 0) return ENGRAM_OK;
    C = s->cfg.recall_c > k ? s->cfg.recall_c : k;
    if (engram_grow((void **)&s->cand, &s->cap_cand, C, sizeof *s->cand) != ENGRAM_OK) return ENGRAM_E_MEM;
    heap = s->cand;
    for (slot = 0; slot < s->n; slot++) {
        engram_cand c;
        if (!(s->flags[slot] & ENGRAM_EPI_LIVE)) continue;
        (void)engram_encq_sig_score(s->q, s->sig + slot * ENGRAM_SIG_WORDS, &c.s);
        c.slot = slot;
        if (hn < C) {
            size_t x = hn++;
            heap[x] = c;
            while (x > 0u && engram_cand_worse(&heap[x], &heap[(x - 1u) / 2u])) {
                engram_cand t = heap[x]; heap[x] = heap[(x - 1u) / 2u]; heap[(x - 1u) / 2u] = t;
                x = (x - 1u) / 2u;
            }
        } else if (engram_cand_worse(&heap[0], &c)) {
            heap[0] = c;
            engram_heap_sift(heap, hn, 0u);
        }
    }
    /* stage 2: exact scores, then an insertion sort of the candidates (C is small) */
    for (i = 0; i < hn; i++) {
        double e = 0.0;
        (void)engram_encq_score(s->q, s->text + s->off[heap[i].slot], s->len[heap[i].slot], &e);
        heap[i].s = e;
    }
    for (i = 1; i < hn; i++) {
        engram_cand c = heap[i];
        j = i;
        while (j > 0u && engram_cand_worse(&heap[j - 1u], &c)) { heap[j] = heap[j - 1u]; j--; }
        heap[j] = c;
    }
    for (i = 0; i < hn && i < k; i++) { hits[i].id = s->id[heap[i].slot]; hits[i].score = heap[i].s; }
    *n_hits = i;
    if (i > 0u) s->recalls[heap[0].slot]++;
    return ENGRAM_OK;
}

/* ---- stats and fingerprint ------------------------------------------------------------------- */
void engram_store_stats_get(const engram_store *s, engram_store_stats *st)
{
    if (!st) return;
    memset(st, 0, sizeof *st);
    if (!s) return;
    st->live = s->live; st->tombstones = s->n - s->live; st->capacity = s->cap_id;
    st->text_bytes = s->live_bytes; st->text_capacity = s->text_cap; st->consolidated = s->consolidated;
    st->next_id = s->next_id; st->adds = s->adds; st->deletes = s->deletes; st->evictions = s->evictions;
    st->refusals = s->refusals; st->compactions = s->compactions;
    st->bytes_resident = sizeof *s + s->cap_id * 8u + s->cap_time * 8u + s->cap_src * 4u + s->cap_flags * 4u +
                         s->cap_rec * 4u + s->cap_off * sizeof(size_t) + s->cap_len * 2u + s->cap_sig * 8u +
                         s->text_cap + sizeof *s->q + s->cap_cand * sizeof *s->cand;
}

uint64_t engram_store_fingerprint(const engram_store *s)
{
    uint64_t h;
    size_t slot;
    if (!s) return 0;
    h = engram_mix2(0x53544F5245ull, s->geometry);
    h = engram_mix2(h, s->next_id);
    h = engram_mix2(h, (uint64_t)s->live);
    for (slot = 0; slot < s->n; slot++) {
        if (!(s->flags[slot] & ENGRAM_EPI_LIVE)) continue;
        h = engram_mix2(h, s->id[slot]);
        h = engram_mix2(h, s->time_ms[slot]);
        h = engram_mix2(h, ((uint64_t)s->source[slot] << 32) | s->flags[slot]);
        h = engram_mix2(h, ((uint64_t)s->recalls[slot] << 32) | s->len[slot]);
        h = engram_mix2(h, engram_hash_bytes(s->text + s->off[slot], s->len[slot], 0u));
        h = engram_mix2(h, engram_hash_bytes(s->sig + slot * ENGRAM_SIG_WORDS,
                                             ENGRAM_SIG_WORDS * sizeof *s->sig, 1u));
    }
    return h;
}
