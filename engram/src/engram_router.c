/* ==================================================================================================
 * engram_router.c -- the router. Contract in engram_router.h.
 * ==================================================================================================
 * LAYOUT. Keys live densely in [0, n): key vectors, ids, bucket of each key, position of each key in
 * its bucket's list. Buckets are growable lists of key indices. An open-addressing table maps id to
 * index (linear probing, BACKWARD-SHIFT deletion -- Knuth 6.4 Algorithm R -- so there are no
 * tombstones to accumulate). Remove moves the last key into the hole: every structure stays dense,
 * and remove never allocates.
 *
 * TRAINING READS KEYS IN ID ORDER. Index order depends on the history of adds and removes; id order
 * does not. So k-means sees the same sequence -- and fits the same centroids -- for the same set of
 * keys however it was assembled, which is what lets the fingerprint be a function of the set.
 * ============================================================================================== */
#include "engram_router.h"
#include "engram_alloc.h"
#include "engram_rng.h"

#include <math.h>
#include <string.h>

typedef struct { uint32_t *a; size_t len, cap; } engram_bucket;

struct engram_router {
    engram_router_cfg cfg;
    size_t            n, cap_key, cap_id, cap_bkt, cap_pos;
    float            *key;                 /* n * dim                                   */
    uint64_t         *id;
    uint32_t         *bkt, *pos;
    unsigned          C;                   /* buckets: 1 when untrained                 */
    int               trained;
    float            *cen;                 /* C * dim when trained                      */
    engram_bucket    *b;
    uint64_t         *hkey;                /* id map: 0 = empty                         */
    uint32_t         *hval;
    size_t            hsize;               /* power of two                              */
    engram_route_hit *heap;                /* search scratch                            */
    size_t            cap_heap;
    engram_route_hit *cheap;               /* the nprobe best centroids                 */
    size_t            cap_cheap;
    uint64_t          adds, removes, trains, searches;
};

void engram_router_cfg_default(engram_router_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->dim = ENGRAM_D;
    cfg->nprobe = 8u;
    cfg->iters = 16u;
    cfg->train_max = 65536u;
    cfg->seed = ENGRAM_ROUTER_SEED;
}

/* A dot product whose association is fixed -- eight lanes, summed pairwise -- so it computes the same
 * bits wherever it runs (R5; -ffp-contract=off forbids fusing the multiply into the add). */
static float engram_rdot(const float *a, const float *b, unsigned d)
{
    float l[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    unsigned i, k;
    for (i = 0; i < d; i += 8u)
        for (k = 0; k < 8u; k++) l[k] += a[i + k] * b[i + k];
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

static int engram_unit_ok(const float *x, unsigned d)
{
    double s = 0.0;
    unsigned i;
    for (i = 0; i < d; i++) {
        double v = (double)x[i];
        if (!(v == v) || v > 1e30 || v < -1e30) return 0;          /* NaN or infinite */
        s += v * v;
    }
    s = sqrt(s);
    return s >= 1.0 - ENGRAM_ROUTER_UNIT_TOL && s <= 1.0 + ENGRAM_ROUTER_UNIT_TOL;
}

/* ---- the id map ------------------------------------------------------------------------------ */
static size_t engram_hhome(const engram_router *r, uint64_t id)
{
    return (size_t)(engram_mix64(id) & (uint64_t)(r->hsize - 1u));
}

/* slot holding id, or (size_t)-1 */
static size_t engram_hfind(const engram_router *r, uint64_t id)
{
    size_t i;
    if (!r->hsize) return (size_t)-1;
    for (i = engram_hhome(r, id);; i = (i + 1u) & (r->hsize - 1u)) {
        if (r->hkey[i] == id) return i;
        if (r->hkey[i] == 0u) return (size_t)-1;
    }
}

static void engram_hput(uint64_t *hk, uint32_t *hv, size_t hsize, uint64_t id, uint32_t v)
{
    size_t i = (size_t)(engram_mix64(id) & (uint64_t)(hsize - 1u));
    while (hk[i] != 0u) i = (i + 1u) & (hsize - 1u);
    hk[i] = id;
    hv[i] = v;
}

/* Backward-shift deletion: every entry after the hole that could live in it moves back. */
static void engram_hdel(engram_router *r, size_t i)
{
    size_t j = i, mask = r->hsize - 1u;
    for (;;) {
        size_t h;
        j = (j + 1u) & mask;
        if (r->hkey[j] == 0u) break;
        h = engram_hhome(r, r->hkey[j]);
        /* the entry at j may move to i unless its home lies cyclically in (i, j] */
        if (i <= j ? (h <= i || h > j) : (h <= i && h > j)) {
            r->hkey[i] = r->hkey[j];
            r->hval[i] = r->hval[j];
            i = j;
        }
    }
    r->hkey[i] = 0u;
    r->hval[i] = 0u;
}

/* ---- open / close ---------------------------------------------------------------------------- */
engram_rc engram_router_open(engram_router **out, const engram_router_cfg *cfg)
{
    engram_router *r;
    engram_router_cfg d;
    if (!out) return ENGRAM_E_ARG;
    *out = NULL;
    if (!cfg) { engram_router_cfg_default(&d); cfg = &d; }
    if (cfg->dim < 8u || cfg->dim > 4096u || cfg->dim % 8u || !cfg->nprobe || !cfg->iters) return ENGRAM_E_ARG;
    r = (engram_router *)engram_malloc(sizeof *r);
    if (!r) return ENGRAM_E_MEM;
    memset(r, 0, sizeof *r);
    r->cfg = *cfg;
    r->C = 1u;
    r->b = (engram_bucket *)engram_calloc(1u, sizeof *r->b);
    if (!r->b) { engram_free(r); return ENGRAM_E_MEM; }
    *out = r;
    return ENGRAM_OK;
}

static void engram_buckets_free(engram_bucket *b, unsigned C)
{
    unsigned i;
    if (!b) return;
    for (i = 0; i < C; i++) engram_free(b[i].a);
    engram_free(b);
}

void engram_router_close(engram_router *r)
{
    if (!r) return;
    engram_free(r->key); engram_free(r->id); engram_free(r->bkt); engram_free(r->pos);
    engram_free(r->cen); engram_buckets_free(r->b, r->C);
    engram_free(r->hkey); engram_free(r->hval); engram_free(r->heap); engram_free(r->cheap);
    engram_free(r);
}

/* ---- add / remove / get ---------------------------------------------------------------------- */
static unsigned engram_nearest(const engram_router *r, const float *x)
{
    unsigned c, best = 0;
    float bs = 0.0f;
    if (!r->trained) return 0u;
    for (c = 0; c < r->C; c++) {
        float s = engram_rdot(x, r->cen + (size_t)c * r->cfg.dim, r->cfg.dim);
        if (c == 0u || s > bs) { bs = s; best = c; }
    }
    return best;
}

engram_rc engram_router_add(engram_router *r, uint64_t id, const float *key)
{
    unsigned tb;
    size_t need = 0, dim;
    engram_bucket *B;
    if (!r || !key || id == 0u) return ENGRAM_E_ARG;
    dim = r->cfg.dim;
    if (!engram_unit_ok(key, r->cfg.dim)) return ENGRAM_E_ARG;
    if (engram_hfind(r, id) != (size_t)-1) return ENGRAM_E_EXISTS;
    if (r->n >= 0xFFFFFFFEu) return ENGRAM_E_FULL;                  /* indices are 32-bit */
    tb = engram_nearest(r, key);
    B = &r->b[tb];
    /* every allocation first; a failure leaves every array's CONTENT as it was */
    need = r->n + 1u;
    if (need > (size_t)-1 / dim) return ENGRAM_E_OVERFLOW;
    if (engram_grow((void **)&r->key, &r->cap_key, need * dim, sizeof *r->key) != ENGRAM_OK ||
        engram_grow((void **)&r->id, &r->cap_id, need, sizeof *r->id) != ENGRAM_OK ||
        engram_grow((void **)&r->bkt, &r->cap_bkt, need, sizeof *r->bkt) != ENGRAM_OK ||
        engram_grow((void **)&r->pos, &r->cap_pos, need, sizeof *r->pos) != ENGRAM_OK ||
        engram_grow((void **)&B->a, &B->cap, B->len + 1u, sizeof *B->a) != ENGRAM_OK)
        return ENGRAM_E_MEM;
    if (2u * need > r->hsize) {                                     /* keep the map at most half full */
        size_t hs = r->hsize ? 2u * r->hsize : 16u, i;
        uint64_t *hk = (uint64_t *)engram_calloc(hs, sizeof *hk);
        uint32_t *hv = (uint32_t *)engram_calloc(hs, sizeof *hv);
        if (!hk || !hv) { engram_free(hk); engram_free(hv); return ENGRAM_E_MEM; }
        for (i = 0; i < r->hsize; i++) if (r->hkey[i]) engram_hput(hk, hv, hs, r->hkey[i], r->hval[i]);
        engram_free(r->hkey); engram_free(r->hval);
        r->hkey = hk; r->hval = hv; r->hsize = hs;
    }
    /* commit: nothing below can fail */
    memcpy(r->key + r->n * dim, key, dim * sizeof *key);
    r->id[r->n] = id;
    r->bkt[r->n] = tb;
    r->pos[r->n] = (uint32_t)B->len;
    B->a[B->len++] = (uint32_t)r->n;
    engram_hput(r->hkey, r->hval, r->hsize, id, (uint32_t)r->n);
    r->n++;
    r->adds++;
    return ENGRAM_OK;
}

engram_rc engram_router_remove(engram_router *r, uint64_t id)
{
    size_t slot, i, last, dim;
    engram_bucket *B;
    uint32_t moved;
    if (!r || id == 0u) return ENGRAM_E_ARG;
    slot = engram_hfind(r, id);
    if (slot == (size_t)-1) return ENGRAM_E_NOTFOUND;
    dim = r->cfg.dim;
    i = r->hval[slot];
    engram_hdel(r, slot);
    /* out of its bucket: the bucket's last entry fills the hole */
    B = &r->b[r->bkt[i]];
    moved = B->a[B->len - 1u];
    B->a[r->pos[i]] = moved;
    r->pos[moved] = r->pos[i];
    B->len--;
    /* the last key fills the hole in the key arrays */
    last = r->n - 1u;
    if (i != last) {
        size_t hs;
        memcpy(r->key + i * dim, r->key + last * dim, dim * sizeof *r->key);
        r->id[i] = r->id[last];
        r->bkt[i] = r->bkt[last];
        r->pos[i] = r->pos[last];
        r->b[r->bkt[i]].a[r->pos[i]] = (uint32_t)i;
        hs = engram_hfind(r, r->id[i]);
        r->hval[hs] = (uint32_t)i;
    }
    r->n--;
    r->removes++;
    return ENGRAM_OK;
}

engram_rc engram_router_get(const engram_router *r, uint64_t id, float *key)
{
    size_t slot;
    if (!r || !key || id == 0u) return ENGRAM_E_ARG;
    slot = engram_hfind(r, id);
    if (slot == (size_t)-1) return ENGRAM_E_NOTFOUND;
    memcpy(key, r->key + (size_t)r->hval[slot] * r->cfg.dim, r->cfg.dim * sizeof *key);
    return ENGRAM_OK;
}

/* ---- training: spherical k-means ------------------------------------------------------------- */
/* Key indices into ascending id order. A heapsort, because qsort takes no context and a static one
 * would make two routers training on two threads race. Ids are distinct: the order is total. */
static void engram_id_sift(uint32_t *a, size_t n, size_t i, const uint64_t *id)
{
    for (;;) {
        size_t l = 2u * i + 1u, rr = l + 1u, m = i;
        uint32_t t;
        if (l < n && id[a[l]] > id[a[m]]) m = l;
        if (rr < n && id[a[rr]] > id[a[m]]) m = rr;
        if (m == i) return;
        t = a[i]; a[i] = a[m]; a[m] = t;
        i = m;
    }
}

static void engram_sort_by_id(uint32_t *a, size_t n, const uint64_t *id)
{
    size_t i;
    if (n < 2u) return;
    for (i = n / 2u; i-- > 0u; ) engram_id_sift(a, n, i, id);
    for (i = n - 1u; i > 0u; i--) {
        uint32_t t = a[0]; a[0] = a[i]; a[i] = t;
        engram_id_sift(a, i, 0u, id);
    }
}

engram_rc engram_router_train(engram_router *r, unsigned C)
{
    size_t n, m, i, dim;
    unsigned c, it;
    uint32_t *order = NULL, *samp, *asg = NULL, *fin = NULL, *nbkt = NULL, *npos = NULL;
    float *cen = NULL, *dist = NULL;
    double *acc = NULL;
    size_t *cnt = NULL;
    engram_bucket *nb = NULL;
    engram_rng rng;
    engram_rc rc = ENGRAM_E_MEM;
    if (!r) return ENGRAM_E_ARG;
    n = r->n;
    dim = r->cfg.dim;
    if (n == 0u) return ENGRAM_E_EMPTY;
    if (C == 0u || C > n) return ENGRAM_E_ARG;
    m = (r->cfg.train_max && n > r->cfg.train_max) ? r->cfg.train_max : n;
    if (m < C) m = C;

    order = (uint32_t *)engram_array(n, sizeof *order);
    asg = (uint32_t *)engram_array(m, sizeof *asg);
    fin = (uint32_t *)engram_array(n, sizeof *fin);
    cen = (float *)engram_array((size_t)C * dim, sizeof *cen);
    dist = (float *)engram_array(m, sizeof *dist);
    acc = (double *)engram_array((size_t)C * dim, sizeof *acc);
    cnt = (size_t *)engram_array(C, sizeof *cnt);
    nbkt = (uint32_t *)engram_array(r->cap_bkt, sizeof *nbkt);
    npos = (uint32_t *)engram_array(r->cap_pos, sizeof *npos);
    nb = (engram_bucket *)engram_calloc(C, sizeof *nb);
    if (!order || !asg || !fin || !cen || !dist || !acc || !cnt || !nbkt || !npos || !nb) goto out;

    /* the keys in id order, and an evenly spaced sample of them */
    for (i = 0; i < n; i++) order[i] = (uint32_t)i;
    engram_sort_by_id(order, n, r->id);
    samp = order;
    if (m < n) {                                   /* in place: order[j] = order[j * n / m] */
        for (i = 0; i < m; i++) order[i] = order[(size_t)(((uint64_t)i * n) / m)];
    }

    /* k-means++ seeding on 1 - cosine */
    engram_rng_seed(&rng, engram_mix2(r->cfg.seed, ((uint64_t)C << 32) ^ (uint64_t)m));
    {
        size_t first = (size_t)engram_rng_below(&rng, m);
        memcpy(cen, r->key + (size_t)samp[first] * dim, dim * sizeof *cen);
    }
    for (i = 0; i < m; i++) {
        float d = 1.0f - engram_rdot(r->key + (size_t)samp[i] * dim, cen, (unsigned)dim);
        dist[i] = d > 0.0f ? d : 0.0f;
    }
    for (c = 1; c < C; c++) {
        double tot = 0.0, pick;
        size_t sel = 0;
        for (i = 0; i < m; i++) tot += (double)dist[i];
        if (tot > 0.0) {
            pick = (double)(engram_rng_u64(&rng) >> 11) * (1.0 / 9007199254740992.0) * tot;
            for (sel = 0; sel + 1u < m; sel++) {
                if (pick < (double)dist[sel] && dist[sel] > 0.0f) break;
                pick -= (double)dist[sel];
            }
            while (sel > 0u && dist[sel] == 0.0f) sel--;       /* never a point already a centre */
        } else sel = c % m;                                   /* every sample on a centre already */
        memcpy(cen + (size_t)c * dim, r->key + (size_t)samp[sel] * dim, dim * sizeof *cen);
        for (i = 0; i < m; i++) {
            float d = 1.0f - engram_rdot(r->key + (size_t)samp[i] * dim, cen + (size_t)c * dim, (unsigned)dim);
            if (d < 0.0f) d = 0.0f;
            if (d < dist[i]) dist[i] = d;
        }
    }

    /* Lloyd iterations: assign, then each centroid becomes its members' normalised mean */
    for (it = 0; it < r->cfg.iters; it++) {
        size_t changed = 0;
        for (i = 0; i < m; i++) {
            const float *x = r->key + (size_t)samp[i] * dim;
            unsigned best = 0;
            float bs = 0.0f;
            for (c = 0; c < C; c++) {
                float s = engram_rdot(x, cen + (size_t)c * dim, (unsigned)dim);
                if (c == 0u || s > bs) { bs = s; best = c; }
            }
            if (it == 0u || asg[i] != best) changed++;
            asg[i] = best;
            dist[i] = 1.0f - bs;
        }
        if (it > 0u && changed == 0u) break;
        memset(acc, 0, (size_t)C * dim * sizeof *acc);
        memset(cnt, 0, C * sizeof *cnt);
        for (i = 0; i < m; i++) {
            const float *x = r->key + (size_t)samp[i] * dim;
            double *a = acc + (size_t)asg[i] * dim;
            size_t d;
            cnt[asg[i]]++;
            for (d = 0; d < dim; d++) a[d] += (double)x[d];
        }
        for (c = 0; c < C; c++) {
            double nn = 0.0;
            size_t d;
            if (cnt[c] == 0u) {                       /* an empty cluster takes the worst-served point */
                size_t w = 0;
                for (i = 1; i < m; i++) if (dist[i] > dist[w]) w = i;
                memcpy(cen + (size_t)c * dim, r->key + (size_t)samp[w] * dim, dim * sizeof *cen);
                dist[w] = 0.0f;
                continue;
            }
            for (d = 0; d < dim; d++) nn += acc[(size_t)c * dim + d] * acc[(size_t)c * dim + d];
            if (!(nn > 0.0)) continue;                /* members cancel exactly: keep the old centre */
            nn = sqrt(nn);
            for (d = 0; d < dim; d++) cen[(size_t)c * dim + d] = (float)(acc[(size_t)c * dim + d] / nn);
        }
    }

    /* every key to its nearest centroid AFTER the last update, so a key always routes to its own
     * bucket at nprobe 1 */
    memset(cnt, 0, C * sizeof *cnt);
    for (i = 0; i < n; i++) {
        const float *x = r->key + i * dim;
        unsigned best = 0;
        float bs = 0.0f;
        for (c = 0; c < C; c++) {
            float s = engram_rdot(x, cen + (size_t)c * dim, (unsigned)dim);
            if (c == 0u || s > bs) { bs = s; best = c; }
        }
        fin[i] = best;
        cnt[best]++;
    }
    for (c = 0; c < C; c++) {                         /* each bucket with a quarter's room to grow */
        size_t cap = cnt[c] + cnt[c] / 4u + 4u;
        nb[c].a = (uint32_t *)engram_array(cap, sizeof *nb[c].a);
        if (!nb[c].a) goto out;
        nb[c].cap = cap;
    }
    /* keys enter their buckets in id order, so bucket contents are a function of the set too */
    for (i = 0; i < n; i++) order[i] = (uint32_t)i;
    engram_sort_by_id(order, n, r->id);
    for (i = 0; i < n; i++) {
        uint32_t k = order[i];
        engram_bucket *B = &nb[fin[k]];
        nbkt[k] = fin[k];
        npos[k] = (uint32_t)B->len;
        B->a[B->len++] = k;
    }

    /* commit */
    engram_free(r->cen);
    engram_buckets_free(r->b, r->C);
    engram_free(r->bkt);
    engram_free(r->pos);
    r->cen = cen; cen = NULL;
    r->b = nb; nb = NULL;
    r->bkt = nbkt; nbkt = NULL;
    r->pos = npos; npos = NULL;
    r->C = C;
    r->trained = 1;
    r->trains++;
    rc = ENGRAM_OK;
out:
    engram_free(order); engram_free(asg); engram_free(fin); engram_free(cen); engram_free(dist);
    engram_free(acc); engram_free(cnt); engram_free(nbkt); engram_free(npos);
    engram_buckets_free(nb, C);
    return rc;
}

/* ---- search ------------------------------------------------------------------------------------ */
/* worse: lower score, or an equal score and a higher id */
static int engram_hit_worse(const engram_route_hit *a, const engram_route_hit *b)
{
    return a->score < b->score || (a->score == b->score && a->id > b->id);
}

static void engram_hit_sift(engram_route_hit *h, size_t n, size_t i)
{
    for (;;) {
        size_t l = 2u * i + 1u, rr = l + 1u, m = i;
        engram_route_hit t;
        if (l < n && engram_hit_worse(&h[l], &h[m])) m = l;
        if (rr < n && engram_hit_worse(&h[rr], &h[m])) m = rr;
        if (m == i) return;
        t = h[i]; h[i] = h[m]; h[m] = t;
        i = m;
    }
}

/* keep the k best in a min-heap whose root is the worst kept */
static void engram_hit_push(engram_route_hit *h, size_t *hn, size_t k, engram_route_hit x)
{
    if (*hn < k) {
        size_t j = (*hn)++;
        h[j] = x;
        while (j > 0u && engram_hit_worse(&h[j], &h[(j - 1u) / 2u])) {
            engram_route_hit t = h[j]; h[j] = h[(j - 1u) / 2u]; h[(j - 1u) / 2u] = t;
            j = (j - 1u) / 2u;
        }
    } else if (engram_hit_worse(&h[0], &x)) {
        h[0] = x;
        engram_hit_sift(h, *hn, 0u);
    }
}

/* the heap into best-first order */
static void engram_hit_drain(engram_route_hit *h, size_t hn, engram_route_hit *out)
{
    size_t i;
    for (i = hn; i > 0u; i--) {
        out[i - 1u] = h[0];
        h[0] = h[i - 1u];
        engram_hit_sift(h, i - 1u, 0u);
    }
}

static engram_rc engram_router_query(engram_router *r, const float *q, size_t k, unsigned nprobe, int exact,
                                     engram_route_hit *hits, size_t *n_hits, engram_route_cost *cost)
{
    size_t hn = 0, scored = 0, i, dim;
    unsigned probe, c, used = 0;
    engram_route_cost local;
    if (!cost) cost = &local;
    memset(cost, 0, sizeof *cost);
    if (n_hits) *n_hits = 0;
    if (!r || !q || !hits || !n_hits || !k) return ENGRAM_E_ARG;
    if (!engram_unit_ok(q, r->cfg.dim)) return ENGRAM_E_ARG;
    dim = r->cfg.dim;
    probe = nprobe ? nprobe : r->cfg.nprobe;
    if (probe > r->C) probe = r->C;
    if (engram_grow((void **)&r->heap, &r->cap_heap, k, sizeof *r->heap) != ENGRAM_OK ||
        engram_grow((void **)&r->cheap, &r->cap_cheap, probe, sizeof *r->cheap) != ENGRAM_OK)
        return ENGRAM_E_MEM;
    r->searches++;
    if (exact || !r->trained) {
        for (i = 0; i < r->n; i++) {
            engram_route_hit x;
            x.id = r->id[i];
            x.score = engram_rdot(q, r->key + i * dim, (unsigned)dim);
            engram_hit_push(r->heap, &hn, k, x);
        }
        scored = r->n;
        used = r->trained ? r->C : 1u;
    } else {
        size_t cn = 0;
        engram_route_hit *cb = r->cheap;
        for (c = 0; c < r->C; c++) {                  /* the nprobe nearest centroids; ties: lower c */
            engram_route_hit x;
            x.id = c;
            x.score = engram_rdot(q, r->cen + (size_t)c * dim, (unsigned)dim);
            engram_hit_push(cb, &cn, probe, x);
        }
        cost->centroids_scored = r->C;
        for (c = 0; c < cn; c++) {
            const engram_bucket *B = &r->b[cb[c].id];
            size_t j;
            for (j = 0; j < B->len; j++) {
                engram_route_hit x;
                uint32_t ki = B->a[j];
                x.id = r->id[ki];
                x.score = engram_rdot(q, r->key + (size_t)ki * dim, (unsigned)dim);
                engram_hit_push(r->heap, &hn, k, x);
            }
            scored += B->len;
        }
        used = (unsigned)cn;
    }
    engram_hit_drain(r->heap, hn, hits);
    *n_hits = hn;
    cost->keys_scored = scored;
    cost->buckets_probed = used;
    return ENGRAM_OK;
}

engram_rc engram_router_search(engram_router *r, const float *q, size_t k, unsigned nprobe,
                               engram_route_hit *hits, size_t *n_hits, engram_route_cost *cost)
{
    return engram_router_query(r, q, k, nprobe, 0, hits, n_hits, cost);
}

engram_rc engram_router_exact(engram_router *r, const float *q, size_t k, engram_route_hit *hits,
                              size_t *n_hits, engram_route_cost *cost)
{
    return engram_router_query(r, q, k, 0u, 1, hits, n_hits, cost);
}

/* ---- stats, fingerprint, invariants -------------------------------------------------------------- */
void engram_router_stats_get(const engram_router *r, engram_router_stats *st)
{
    unsigned c;
    double sq = 0.0;
    if (!st) return;
    memset(st, 0, sizeof *st);
    if (!r) return;
    st->keys = r->n;
    st->buckets = r->C;
    st->trained = r->trained;
    for (c = 0; c < r->C; c++) {
        size_t len = r->b[c].len;
        if (!len) st->empty_buckets++;
        if (len > st->largest_bucket) st->largest_bucket = len;
        sq += (double)len * (double)len;
    }
    st->imbalance = r->n ? sq * (double)r->C / ((double)r->n * (double)r->n) : 1.0;
    st->adds = r->adds; st->removes = r->removes; st->trains = r->trains; st->searches = r->searches;
    st->bytes_resident = sizeof *r + r->cap_key * sizeof(float) + r->cap_id * 8u + r->cap_bkt * 4u +
                         r->cap_pos * 4u + (r->trained ? (size_t)r->C * r->cfg.dim * sizeof(float) : 0u) +
                         r->C * sizeof *r->b + r->hsize * 12u + r->cap_heap * sizeof *r->heap +
                         r->cap_cheap * sizeof *r->cheap;
    for (c = 0; c < r->C; c++) st->bytes_resident += r->b[c].cap * sizeof(uint32_t);
}

uint64_t engram_router_fingerprint(const engram_router *r)
{
    uint64_t h, sum = 0, x = 0;
    size_t i;
    if (!r) return 0u;
    h = engram_mix2(0x524F55544552ull, ((uint64_t)r->cfg.dim << 32) | r->cfg.nprobe);
    h = engram_mix2(h, ((uint64_t)r->cfg.iters << 32) | (uint64_t)r->trained);
    h = engram_mix2(h, (uint64_t)r->cfg.train_max);
    h = engram_mix2(h, r->cfg.seed);
    h = engram_mix2(h, ((uint64_t)r->C << 32) | (uint64_t)r->n);
    if (r->trained) h = engram_mix2(h, engram_hash_bytes(r->cen, (size_t)r->C * r->cfg.dim * sizeof *r->cen, 7u));
    for (i = 0; i < r->n; i++) {                      /* the key set: order-free by construction */
        uint64_t k = engram_mix2(r->id[i], engram_hash_bytes(r->key + i * r->cfg.dim, r->cfg.dim * sizeof *r->key, 3u));
        k = engram_mix2(k, (uint64_t)r->bkt[i]);
        sum += k;
        x ^= engram_mix64(k);
    }
    return engram_mix2(engram_mix2(h, sum), x);
}

engram_rc engram_router_check(const engram_router *r)
{
    size_t i, total = 0, used = 0;
    unsigned c;
    if (!r) return ENGRAM_E_ARG;
    if (!r->b || r->C == 0u || (r->trained && !r->cen) || (!r->trained && r->C != 1u)) return ENGRAM_E_INTERNAL;
    for (c = 0; c < r->C; c++) {
        const engram_bucket *B = &r->b[c];
        size_t j;
        if (B->len > B->cap) return ENGRAM_E_INTERNAL;
        for (j = 0; j < B->len; j++) {
            uint32_t k = B->a[j];
            if (k >= r->n || r->bkt[k] != c || r->pos[k] != j) return ENGRAM_E_INTERNAL;
        }
        total += B->len;
    }
    if (total != r->n) return ENGRAM_E_INTERNAL;
    for (i = 0; i < r->n; i++) {
        size_t s;
        if (r->id[i] == 0u) return ENGRAM_E_INTERNAL;
        s = engram_hfind(r, r->id[i]);
        if (s == (size_t)-1 || r->hval[s] != i) return ENGRAM_E_INTERNAL;
    }
    for (i = 0; i < r->hsize; i++) {
        if (!r->hkey[i]) continue;
        used++;
        if (r->hval[i] >= r->n || r->id[r->hval[i]] != r->hkey[i]) return ENGRAM_E_INTERNAL;
    }
    if (used != r->n) return ENGRAM_E_INTERNAL;
    if (r->trained && r->C > 1u && r->n > 0u) {       /* every key sits in its nearest centroid's bucket
                                                         only right after training; inserts keep it too */
        for (i = 0; i < r->n; i++)
            if (engram_nearest(r, r->key + i * r->cfg.dim) != r->bkt[i]) return ENGRAM_E_INTERNAL;
    }
    return ENGRAM_OK;
}
