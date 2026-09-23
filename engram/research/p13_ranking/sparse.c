/* sparse.c -- does EXPAND-THEN-SPARSIFY (engram.h: ENGRAM_EPI_E / _K / _FANIN) earn a role in the store?
 * The P1.3 spec asked for it, the P1.3 plan said to measure it before building it, and this is that
 * measurement, on the shared lab protocol (lab.c): 21,811 chunks, 300 dev sources x 12 fragment cues.
 *
 * The code: the chunk's dense unit vector (engram_encode, D = 512) through a fixed pseudo-random +-1
 * matrix to E = 2048 outputs, each output reading F of the D inputs (F = 0: all of them), then the
 * K = 32 outputs of largest |value| kept with their signs. Similarity: agreeing signs on shared
 * winners minus disagreeing ones, over K -- the cosine of two +-1 K-sparse vectors.
 *
 * Asked, against the two things the store already has:
 *   candidate recall@C   is the source in the top C? (the signature: 0.9900 / 0.9953 / 0.9983)
 *   rescue               of the cues whose source the signature's top 50 MISSES, how many does the
 *                        sparse code's top 50 hold? (a first stage it could join rather than replace)
 * usage: ENGRAM_TEST_DATA=. ./sparse N dev|test nsrc */
#define LAB_NO_MAIN
#include "lab.c"

#define SE ENGRAM_EPI_E
#define SK ENGRAM_EPI_K
#define NFAN 4
static const unsigned FAN[NFAN] = { 0u, 16u, 64u, 256u };

/* one code: K signed winners, sorted by index; entry = (index << 1) | negative */
static void sparse_code(const float *x, unsigned fan, uint32_t *code)
{
    static float y[SE];
    static uint32_t idx[SE];
    unsigned e, d, k;
    for (e = 0; e < SE; e++) {
        float acc = 0.0f;
        if (fan == 0u) {
            for (d = 0; d < ENGRAM_D; d++) {
                uint64_t h = engram_mix2(0x5A125Eull + e, d);
                acc += (h >> 63) ? -x[d] : x[d];
            }
        } else {
            for (k = 0; k < fan; k++) {
                uint64_t h = engram_mix2(0xFA41Eull + e, k);
                d = (unsigned)((h & 0xFFFFFFFFull) % ENGRAM_D);
                acc += (h >> 63) ? -x[d] : x[d];
            }
        }
        y[e] = acc;
        idx[e] = e;
    }
    /* partial selection: the K largest |y| (ties: lower index) */
    for (k = 0; k < SK; k++) {
        unsigned best = k;
        for (e = k + 1u; e < SE; e++) {
            float a = y[idx[e]] < 0 ? -y[idx[e]] : y[idx[e]], b = y[idx[best]] < 0 ? -y[idx[best]] : y[idx[best]];
            if (a > b || (a == b && idx[e] < idx[best])) best = e;
        }
        { uint32_t t = idx[k]; idx[k] = idx[best]; idx[best] = t; }
    }
    for (k = 0; k < SK; k++) code[k] = (idx[k] << 1) | (y[idx[k]] < 0.0f ? 1u : 0u);
    for (k = 1; k < SK; k++) {                         /* sort by index for the merge */
        uint32_t v = code[k];
        unsigned j = k;
        while (j > 0 && (code[j - 1] >> 1) > (v >> 1)) { code[j] = code[j - 1]; j--; }
        code[j] = v;
    }
}

static int sparse_sim(const uint32_t *a, const uint32_t *b)
{
    unsigned i = 0, j = 0;
    int s = 0;
    while (i < SK && j < SK) {
        uint32_t ia = a[i] >> 1, ib = b[j] >> 1;
        if (ia == ib) { s += ((a[i] ^ b[j]) & 1u) ? -1 : 1; i++; j++; }
        else if (ia < ib) i++; else j++;
    }
    return s;
}

static int int_desc(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x < y) - (x > y); }

int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j;
    int test;
    unsigned f, k;
    float *mat, q[ENGRAM_D];
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    uint32_t *codes, qc[SK];
    int *ss, *srt;
    double *sd, *sds, *s1, *s1s;
    unsigned rec[NFAN][NC], r1[NFAN], drec[NC], srec[NC], rescue[NFAN], missed = 0, dr1 = 0;
    if (argc < 4) return 2;
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    engram_enc_cfg_default(&cfg);
    mat = index_build(&cfg, ENGRAM_D, &c);
    sig = sig_build(&cfg, &c);
    codes = (uint32_t *)engram_array(c.n * NFAN * SK, sizeof *codes);
    ss = (int *)engram_array(c.n, sizeof *ss); srt = (int *)engram_array(c.n, sizeof *srt);
    sd = (double *)engram_array(c.n, sizeof *sd); sds = (double *)engram_array(c.n, sizeof *sds);
    s1 = (double *)engram_array(c.n, sizeof *s1); s1s = (double *)engram_array(c.n, sizeof *s1s);
    memset(rec, 0, sizeof rec); memset(r1, 0, sizeof r1); memset(drec, 0, sizeof drec); memset(srec, 0, sizeof srec);
    memset(rescue, 0, sizeof rescue);
    for (j = 0; j < c.n; j++)
        for (f = 0; f < NFAN; f++) sparse_code(mat + j * ENGRAM_D, FAN[f], codes + (j * NFAN + f) * SK);
    fprintf(stderr, "codes built for %lu chunks\n", (unsigned long)c.n);
    lab_make_queries(&c, test, nsrc, &Q);
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t];
        int sig_in50;
        if (engram_encode(&cfg, Q.txt[t], Q.len[t], q, NULL) != ENGRAM_OK) continue;
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
        /* the signature and the dense vector, as references */
        for (j = 0; j < c.n; j++) { engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &s1[j]); sd[j] = fdot(q, mat + j * ENGRAM_D, ENGRAM_D); }
        memcpy(s1s, s1, c.n * sizeof *s1); qsort(s1s, c.n, sizeof *s1s, double_desc);
        memcpy(sds, sd, c.n * sizeof *sd); qsort(sds, c.n, sizeof *sds, double_desc);
        for (k = 0; k < NC; k++) {
            srec[k] += s1[i] >= s1s[CS[k] - 1u];
            drec[k] += sd[i] >= sds[CS[k] - 1u];
        }
        dr1 += sd[i] >= sds[0];
        sig_in50 = s1[i] >= s1s[49];
        missed += !sig_in50;
        for (f = 0; f < NFAN; f++) {
            sparse_code(q, FAN[f], qc);
            for (j = 0; j < c.n; j++) ss[j] = sparse_sim(qc, codes + (j * NFAN + f) * SK);
            memcpy(srt, ss, c.n * sizeof *ss); qsort(srt, c.n, sizeof *srt, int_desc);
            for (k = 0; k < NC; k++) rec[f][k] += ss[i] >= srt[CS[k] - 1u];
            r1[f] += ss[i] >= srt[0] && srt[0] > srt[1];   /* strictly first: ties are not wins */
            if (!sig_in50 && ss[i] >= srt[49]) rescue[f]++;
        }
    }
    {   /* POSITIVE CONTROL (R6 both ways): a chunk's OWN text, and the chunk with 2 typos, must find it
         * -- if the code cannot do that, the numbers above measure a bug, not the method */
        unsigned own[NFAN], typo2[NFAN], nctl = 0;
        static uint32_t in[MAXCP], outc[MAXCP];
        static char tbuf[MAXB];
        memset(own, 0, sizeof own); memset(typo2, 0, sizeof typo2);
        for (t = (size_t)test; t < 2u * nsrc && nctl < 300u; t += 2u) {
            size_t i = (t * 7919u + 13u) % c.n, n, m, tl;
            alphabet ab; static uint32_t al[MAXCP]; typo_arg ta; engram_rng r;
            n = decode_all(c.line[i], c.len[i], in, MAXCP);
            ab.a = al; ab.n = 0;
            for (k = 0; k < n; k++) if (is_letter(in[k])) al[ab.n++] = in[k];
            if (!ab.n) { al[0] = 'e'; ab.n = 1; }
            ta.k = 2; ta.ab = &ab;
            engram_rng_seed(&r, engram_mix2(0xC0A7u, (uint64_t)i));
            m = v_typos(&r, in, n, outc, MAXCP, &ta);
            tl = encode_all(outc, m, tbuf, sizeof tbuf);
            nctl++;
            for (f = 0; f < NFAN; f++) {
                unsigned pass;
                for (pass = 0; pass < 2u; pass++) {
                    const char *qt = pass ? tbuf : c.line[i];
                    size_t ql = pass ? tl : c.len[i];
                    if (engram_encode(&cfg, qt, ql, q, NULL) != ENGRAM_OK) continue;
                    sparse_code(q, FAN[f], qc);
                    for (j = 0; j < c.n; j++) ss[j] = sparse_sim(qc, codes + (j * NFAN + f) * SK);
                    memcpy(srt, ss, c.n * sizeof *ss); qsort(srt, c.n, sizeof *srt, int_desc);
                    if (ss[i] >= srt[0]) { if (pass) typo2[f]++; else own[f]++; }
                }
            }
        }
        for (f = 0; f < NFAN; f++)
            printf("  control fanin=%-3u  own text first %.4f   own text + 2 typos first %.4f   (%u chunks)\n",
                   FAN[f], (double)own[f] / nctl, (double)typo2[f] / nctl, nctl);
    }
    printf("N=%lu %s, %lu cues (ties at the C-th place count as inside -- generous to every method)\n",
           (unsigned long)c.n, test ? "test" : "dev", (unsigned long)Q.n);
    printf("  %-26s", "candidate recall @");
    for (k = 0; k < NC; k++) printf("  C=%-4u", CS[k]);
    printf("  first\n");
    printf("  %-26s", "signature (the store)");
    for (k = 0; k < NC; k++) printf("  %.4f", (double)srec[k] / Q.n);
    printf("\n  %-26s", "dense 512");
    for (k = 0; k < NC; k++) printf("  %.4f", (double)drec[k] / Q.n);
    printf("  %.4f\n", (double)dr1 / Q.n);
    for (f = 0; f < NFAN; f++) {
        char name[64];
        snprintf(name, sizeof name, "sparse E=%u K=%u fanin=%u", SE, SK, FAN[f]);
        printf("  %-26s", name);
        for (k = 0; k < NC; k++) printf("  %.4f", (double)rec[f][k] / Q.n);
        printf("  %.4f   rescues %u of the %u the signature's top 50 missed\n", (double)r1[f] / Q.n, rescue[f], missed);
    }
    return 0;
}
