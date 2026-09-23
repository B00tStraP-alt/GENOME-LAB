/* p13lab/lab.c -- THE SHARED P1.3 MEASUREMENT HARNESS. Every approach extends a copy of this file, so
 * every number in the P1.3 decision comes from the same queries, split and metrics.
 *
 * PROTOCOL (fixed before any approach was tried):
 *   store      the first N chunks of corpus_scale.txt (21,811 chunks <= 480 bytes, en/fr/de/sv/zh,
 *              shuffled so any prefix is a mixed store)
 *   sources    t = 0 .. 2*nsrc-1, chunk i = (t * 7919 + 13) mod N; DEV = even t, TEST = odd t.
 *              Choices (weights, C, parameters) are made on DEV only; TEST is reported, never tuned on.
 *   queries    12 per source: 5/8/10/15% fragments from a word boundary x 0/1/2 typos, typo letters
 *              drawn from the source chunk's own letters (engram test generators, seeded per arm+chunk)
 *   metrics    r1_cascade[C]  source ranked first after stage 1 (top C) + stage 2
 *              s1_recall[C]   source inside the stage-1 top C
 *              r1_exhaustive  exact score over the whole store (reference), when --exh
 *              ms_stage1      mean stage-1 time per query
 *   baseline   stage 1 = signature containment (engram_encq_sig_score), stage 2 = exact
 *              Bhattacharyya score (engram_encq_score) -- the shipped P1.2 cascade.
 *
 * build:  ./build.sh lab.c lab        run:  ENGRAM_TEST_DATA=. ./lab N dev|test nsrc [--exh]
 */
#define main test_enc_main_unused
#include "../../test/test_enc.c"
#undef main

static const unsigned CS[] = { 10, 50, 200 };
#define NC 3

typedef struct { size_t n, *src, *len; char **txt; } lab_queries;

static void lab_make_queries(const corpus *c, int test, size_t nsrc, lab_queries *Q)
{
    static uint32_t in[MAXCP], out[MAXCP], al[MAXCP];
    static char txt[MAXB];
    size_t t, k, a;
    Q->n = 0;
    Q->src = (size_t *)engram_array(nsrc * NARM, sizeof *Q->src);
    Q->len = (size_t *)engram_array(nsrc * NARM, sizeof *Q->len);
    Q->txt = (char **)engram_array(nsrc * NARM, sizeof *Q->txt);
    for (t = (size_t)test; t < 2u * nsrc; t += 2u) {
        size_t i = (t * 7919u + 13u) % c->n, n = decode_all(c->line[i], c->len[i], in, MAXCP);
        alphabet ab;
        ab.a = al; ab.n = 0;
        for (k = 0; k < n; k++) if (is_letter(in[k])) al[ab.n++] = in[k];
        if (!ab.n) { al[0] = 'e'; ab.n = 1; }
        for (a = 0; a < NARM; a++) {
            ft_arg fa;
            engram_rng r;
            size_t m, tl;
            fa.pct = ARM_PCT[a % 4u]; fa.k = (unsigned)(a / 4u); fa.ab = &ab;
            engram_rng_seed(&r, engram_mix2(0x5CA1Eu + a, (uint64_t)i));
            m = v_fragtypo(&r, in, n, out, MAXCP, &fa);
            tl = encode_all(out, m, txt, sizeof txt);
            Q->src[Q->n] = i; Q->len[Q->n] = tl;
            Q->txt[Q->n] = (char *)engram_malloc(tl + 1u);
            memcpy(Q->txt[Q->n], txt, tl + 1u);
            Q->n++;
        }
    }
}

/* Exhaustive exact reference by the kernel's symmetry (proven in test_enc T14): each chunk's table is
 * built once and the short queries stream through it. hit[t] = 1 when no other chunk out-scores the
 * source. Chunks <= 480 bytes always fit a table. */
static unsigned lab_exhaustive(const engram_enc_cfg *cfg, const corpus *c, const lab_queries *Q)
{
    engram_encq *tab = (engram_encq *)engram_malloc(sizeof *tab);
    double *src = (double *)engram_array(Q->n, sizeof *src);
    uint8_t *beaten = (uint8_t *)engram_array(Q->n, 1u);
    size_t t, j;
    unsigned pass, hits = 0;
    memset(beaten, 0, Q->n);
    for (pass = 0; pass < 2u; pass++)
        for (j = 0; j < c->n; j++) {
            if (engram_encq_build(tab, cfg, c->line[j], c->len[j]) != ENGRAM_OK) continue;
            for (t = 0; t < Q->n; t++) {
                double e = 0.0;
                if (pass == 0u ? Q->src[t] != j : (Q->src[t] == j || beaten[t])) continue;
                engram_encq_score(tab, Q->txt[t], Q->len[t], &e);
                if (pass == 0u) src[t] = e; else if (e > src[t]) beaten[t] = 1;
            }
        }
    for (t = 0; t < Q->n; t++) hits += !beaten[t];
    engram_free(tab); engram_free(src); engram_free(beaten);
    return hits;
}

#ifndef LAB_NO_MAIN
int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j;
    int test, exh;
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    double *ss, *srt, t0, ts1 = 0.0;
    unsigned k, s1[NC] = { 0 }, cas[NC] = { 0 }, hx = 0;
    if (argc < 4) { fprintf(stderr, "usage: lab N dev|test nsrc [--exh]\n"); return 2; }
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    exh = argc > 4 && strcmp(argv[4], "--exh") == 0;
    if (!corpus_load(&all, "scale")) { fprintf(stderr, "corpus_scale.txt not found\n"); return 2; }
    c = all; if (N < c.n) c.n = N;
    engram_enc_cfg_default(&cfg);
    sig = sig_build(&cfg, &c);
    ss = (double *)engram_array(c.n, sizeof *ss);
    srt = (double *)engram_array(c.n, sizeof *srt);
    lab_make_queries(&c, test, nsrc, &Q);
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t];
        double e_src, e;
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
        t0 = (double)engram_now_ns();
        for (j = 0; j < c.n; j++) engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &ss[j]);
        ts1 += (double)engram_now_ns() - t0;
        engram_encq_score(eq, c.line[i], c.len[i], &e_src);
        memcpy(srt, ss, c.n * sizeof *ss);
        qsort(srt, c.n, sizeof *srt, double_desc);
        for (k = 0; k < NC; k++) {
            double thr = srt[(CS[k] < c.n ? CS[k] : c.n) - 1u];
            int beaten = 0;
            if (ss[i] < thr) continue;
            s1[k]++;
            for (j = 0; j < c.n && !beaten; j++)
                if (j != i && ss[j] >= thr) {
                    engram_encq_score(eq, c.line[j], c.len[j], &e);
                    if (e > e_src) beaten = 1;
                }
            cas[k] += !beaten;
        }
    }
    if (exh) hx = lab_exhaustive(&cfg, &c, &Q);
    printf("{\"approach\": \"baseline\", \"N\": %lu, \"split\": \"%s\", \"queries\": %lu, \"ms_stage1\": %.3f",
           (unsigned long)c.n, test ? "test" : "dev", (unsigned long)Q.n, ts1 / (double)Q.n / 1e6);
    for (k = 0; k < NC; k++)
        printf(", \"s1_recall_%u\": %.4f, \"r1_cascade_%u\": %.4f", CS[k], (double)s1[k] / (double)Q.n,
               CS[k], (double)cas[k] / (double)Q.n);
    if (exh) printf(", \"r1_exhaustive\": %.4f", (double)hx / (double)Q.n);
    printf("}\n");
    return 0;
}
#endif /* LAB_NO_MAIN */
