/* ivf.c -- P1.4 lab: an IVF coarse quantiser (spherical k-means, one bucket per key) over the dense
 * vectors of the 21,811 scale chunks, measured two ways before anything is built:
 *
 *   ANN      the router's own job: of the EXHAUSTIVE dense top 10 for a query, how many does the
 *            IVF top 10 hold, when it scans only the nprobe nearest buckets? And how many keys did
 *            it actually touch (counted, not estimated)?
 *   STORE    could the router make the episodic store's stage 1 sub-linear? The store scans every
 *            signature (candidate recall of the source 0.9953 at C = 50). Here the signature scan
 *            covers only the probed buckets: does the source still reach the top 50?
 *
 * Cues: the lab's 3,600 dev fragment cues (lab.c), and a near-copy control (the chunk with two
 * typos). k-means: deterministic k-means++ seeding, 12 Lloyd iterations, centroids renormalised.
 * usage: ENGRAM_TEST_DATA=<dir with corpus_scale.txt> ./ivf N nsrc */
#define LAB_NO_MAIN
#include "../p13_ranking/lab.c"

#define NCS 3
static const unsigned CLIST[NCS] = { 64u, 148u, 512u };
#define NNP 7
static const unsigned NPROBE[NNP] = { 1u, 2u, 4u, 8u, 16u, 32u, 64u };

typedef struct { float s; uint32_t i; } fs_t;
static int fs_desc(const void *a, const void *b)
{
    const fs_t *x = (const fs_t *)a, *y = (const fs_t *)b;
    if (x->s != y->s) return x->s < y->s ? 1 : -1;
    return x->i < y->i ? -1 : x->i > y->i;
}

/* spherical k-means; returns centroids (C x D) and assign[N] */
static float *kmeans(const float *X, size_t N, unsigned C, uint32_t *assign, uint64_t seed)
{
    float *cen = (float *)engram_array((size_t)C * ENGRAM_D, sizeof *cen);
    double *acc = (double *)engram_array((size_t)C * ENGRAM_D, sizeof *acc);
    float *best = (float *)engram_array(N, sizeof *best);
    size_t *cnt = (size_t *)engram_array(C, sizeof *cnt);
    engram_rng r;
    unsigned c, it, d;
    size_t i;
    engram_rng_seed(&r, seed);
    /* k-means++ on 1 - cos */
    i = (size_t)engram_rng_below(&r, N);
    memcpy(cen, X + i * ENGRAM_D, ENGRAM_D * sizeof *cen);
    for (i = 0; i < N; i++) best[i] = 1.0f - fdot(X + i * ENGRAM_D, cen, ENGRAM_D);
    for (c = 1; c < C; c++) {
        double tot = 0.0, pick;
        for (i = 0; i < N; i++) tot += best[i] > 0.0f ? best[i] : 0.0f;
        pick = (double)(engram_rng_u64(&r) >> 11) / 9007199254740992.0 * tot;
        for (i = 0; i < N - 1u; i++) { double w = best[i] > 0.0f ? best[i] : 0.0f; if (pick < w) break; pick -= w; }
        memcpy(cen + (size_t)c * ENGRAM_D, X + i * ENGRAM_D, ENGRAM_D * sizeof *cen);
        for (i = 0; i < N; i++) {
            float dd = 1.0f - fdot(X + i * ENGRAM_D, cen + (size_t)c * ENGRAM_D, ENGRAM_D);
            if (dd < best[i]) best[i] = dd;
        }
    }
    for (it = 0; it < 12u; it++) {
        size_t moved = 0;
        for (i = 0; i < N; i++) {
            float bs = -2.0f; uint32_t bc = 0;
            for (c = 0; c < C; c++) { float s = fdot(X + i * ENGRAM_D, cen + (size_t)c * ENGRAM_D, ENGRAM_D); if (s > bs) { bs = s; bc = c; } }
            if (it == 0 || assign[i] != bc) moved++;
            assign[i] = bc;
        }
        memset(acc, 0, (size_t)C * ENGRAM_D * sizeof *acc); memset(cnt, 0, C * sizeof *cnt);
        for (i = 0; i < N; i++) { cnt[assign[i]]++; for (d = 0; d < ENGRAM_D; d++) acc[(size_t)assign[i] * ENGRAM_D + d] += X[i * ENGRAM_D + d]; }
        for (c = 0; c < C; c++) {
            double nn = 0.0;
            if (!cnt[c]) continue;                         /* an empty cluster keeps its old centroid */
            for (d = 0; d < ENGRAM_D; d++) nn += acc[(size_t)c * ENGRAM_D + d] * acc[(size_t)c * ENGRAM_D + d];
            nn = sqrt(nn);
            for (d = 0; d < ENGRAM_D; d++) cen[(size_t)c * ENGRAM_D + d] = (float)(acc[(size_t)c * ENGRAM_D + d] / nn);
        }
        if (it > 0 && moved == 0) break;
    }
    engram_free(acc); engram_free(best); engram_free(cnt);
    return cen;
}

int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j;
    unsigned ci, pi, k;
    float *mat, q[ENGRAM_D];
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    fs_t *ex, *cs;
    double *s1;
    if (argc < 3) return 2;
    N = (size_t)atoll(argv[1]); nsrc = (size_t)atoll(argv[2]);
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    engram_enc_cfg_default(&cfg);
    mat = index_build(&cfg, ENGRAM_D, &c);
    sig = sig_build(&cfg, &c);
    lab_make_queries(&c, 0, nsrc, &Q);
    ex = (fs_t *)engram_array(c.n, sizeof *ex);
    cs = (fs_t *)engram_array(1024, sizeof *cs);
    s1 = (double *)engram_array(c.n, sizeof *s1);
    printf("N=%lu, %lu dev fragment cues; full signature scan holds the source in its top 50 for 0.9953\n",
           (unsigned long)c.n, (unsigned long)Q.n);
    for (ci = 0; ci < NCS; ci++) {
        unsigned C = CLIST[ci];
        uint32_t *assign = (uint32_t *)engram_array(c.n, sizeof *assign);
        float *cen;
        size_t *bsize = (size_t *)engram_array(C, sizeof *bsize), bmax = 0, bempty = 0;
        double ann[NNP] = { 0 }, probed[NNP] = { 0 }, store[NNP] = { 0 }, self1 = 0;
        double t0 = (double)engram_now_ns();
        cen = kmeans(mat, c.n, C, assign, 0xC0A25Eull);
        memset(bsize, 0, C * sizeof *bsize);
        for (j = 0; j < c.n; j++) bsize[assign[j]]++;
        for (k = 0; k < C; k++) { if (bsize[k] > bmax) bmax = bsize[k]; bempty += !bsize[k]; }
        printf("C=%u: k-means %.1f s, largest bucket %lu (mean %.0f), %lu empty\n", C,
               ((double)engram_now_ns() - t0) / 1e9, (unsigned long)bmax, (double)c.n / C, (unsigned long)bempty);
        for (t = 0; t < Q.n; t++) {
            size_t i = Q.src[t];
            float thr10;
            if (engram_encode(&cfg, Q.txt[t], Q.len[t], q, NULL) != ENGRAM_OK) continue;
            if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
            for (j = 0; j < c.n; j++) { ex[j].s = fdot(q, mat + j * ENGRAM_D, ENGRAM_D); ex[j].i = (uint32_t)j; }
            qsort(ex, c.n, sizeof *ex, fs_desc);
            thr10 = ex[9].s;
            for (k = 0; k < C; k++) { cs[k].s = fdot(q, cen + (size_t)k * ENGRAM_D, ENGRAM_D); cs[k].i = k; }
            qsort(cs, C, sizeof *cs, fs_desc);
            for (j = 0; j < c.n; j++) engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &s1[j]);
            for (pi = 0; pi < NNP; pi++) {
                unsigned np = NPROBE[pi] < C ? NPROBE[pi] : C, got = 0, b;
                size_t scanned = 0, above = 0;
                static uint8_t inp[1024];
                memset(inp, 0, C);
                for (b = 0; b < np; b++) { inp[cs[b].i] = 1; scanned += bsize[cs[b].i]; }
                for (k = 0; k < 10u; k++) got += inp[assign[ex[k].i]];   /* exhaustive top 10 inside probed buckets */
                (void)thr10;
                ann[pi] += got / 10.0;
                probed[pi] += (double)scanned / (double)c.n;
                if (inp[assign[i]]) {                                      /* store: source in probed, rank < 50 */
                    for (j = 0; j < c.n; j++) if (j != i && inp[assign[j]] && s1[j] > s1[i]) above++;
                    store[pi] += above < 50u;
                }
            }
        }
        /* self-routing control: a key's own vector must land in its own bucket at nprobe 1 */
        for (j = 0; j < c.n; j++) {
            float bs = -2.0f; uint32_t bc = 0;
            for (k = 0; k < C; k++) { float s = fdot(mat + j * ENGRAM_D, cen + (size_t)k * ENGRAM_D, ENGRAM_D); if (s > bs) { bs = s; bc = k; } }
            self1 += bc == assign[j];
        }
        printf("  self-routing (own vector -> own bucket) %.4f\n", self1 / c.n);
        for (pi = 0; pi < NNP; pi++)
            printf("  nprobe %3u  probed %.3f of keys  ANN recall@10 %.4f  store: source in top 50 %.4f\n",
                   NPROBE[pi], probed[pi] / Q.n, ann[pi] / Q.n, store[pi] / Q.n);
        engram_free(assign); engram_free(cen); engram_free(bsize);
    }
    return 0;
}
