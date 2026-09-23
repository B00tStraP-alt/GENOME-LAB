/* solo/rarity.c -- rarity (IDF) in the P1.3 cascade, measured on the shared lab protocol.
 * One pass per query computes, on the SAME stage-1 candidates, every stage-2 scoring function, so the
 * functions differ only in how they rank. Stage 1 is either plain signature containment (baseline) or
 * IDF-weighted containment. Exploratory: float log for IDF (a deterministic form comes after, if IDF
 * earns its place).
 * usage: ENGRAM_TEST_DATA=. ./rarity N dev|test nsrc */
#define LAB_NO_MAIN
#include "lab.c"
#include <math.h>

/* ---- df: an open-addressing table of feature id -> number of chunks containing it ------------ */
static uint64_t *df_key;
static uint32_t *df_val;
static size_t    df_mask;

static uint32_t *df_slot(uint64_t k, int insert)
{
    size_t i = (size_t)(engram_mix64(k) & df_mask);
    for (;;) {
        if (df_key[i] == k) return &df_val[i];
        if (df_key[i] == 0u) {
            if (!insert) return NULL;
            df_key[i] = k; df_val[i] = 0; return &df_val[i];
        }
        i = (i + 1u) & df_mask;
    }
}

static uint32_t df_of(uint64_t k) { uint32_t *v = df_slot(k, 0); return v ? *v : 0u; }

static double NDOC;
static double idf_classic(uint32_t df) { return log((NDOC + 1.0) / ((double)df + 0.5)); }
static double idf_bm25(uint32_t df) { return log(1.0 + (NDOC - (double)df + 0.5) / ((double)df + 0.5)); }

#define NS2 12
static const char *S2NAME[NS2] = {
    "bhatt (baseline)", "bhatt, stage-1 idf", "containment-idf", "bhatt-idf",
    "bm25 k1=0.5 b=0", "bm25 k1=1.2 b=0", "bm25 k1=1.2 b=0.3", "bm25 k1=1.2 b=0.75",
    "bm25 k1=2 b=0.3", "stage-1 idf score alone", "bhatt x cont-idf", "bm25 k1=1.2 b=0.3 x bhatt"
};

int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j, o;
    int test;
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    double *s1p, *s1i, *srt, *dmass, *didfmass, avgdl = 0.0;
    unsigned k, s1c[2][NC], hit[2][NC][NS2];
    memset(s1c, 0, sizeof s1c); memset(hit, 0, sizeof hit);
    if (argc < 4) return 2;
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    NDOC = (double)c.n;
    engram_enc_cfg_default(&cfg);
    sig = sig_build(&cfg, &c);
    /* df over the store: every distinct feature of every chunk (the query never contributes) */
    df_mask = (1u << 24) - 1u;
    df_key = (uint64_t *)calloc(df_mask + 1u, sizeof *df_key);
    df_val = (uint32_t *)calloc(df_mask + 1u, sizeof *df_val);
    dmass = (double *)engram_array(c.n, sizeof *dmass);
    didfmass = (double *)engram_array(c.n, sizeof *didfmass);
    for (j = 0; j < c.n; j++) {
        if (engram_encq_build(eq, &cfg, c.line[j], c.len[j]) != ENGRAM_OK) { dmass[j] = 0; continue; }
        dmass[j] = eq->qmass; avgdl += eq->qmass;
        for (o = 0; o < eq->n; o++) (*df_slot(eq->key[eq->occ[o]], 1))++;
    }
    avgdl /= (double)c.n;
    for (j = 0; j < c.n; j++) {                        /* each doc's idf-weighted mass (bhatt-idf norm) */
        double m = 0.0;
        if (engram_encq_build(eq, &cfg, c.line[j], c.len[j]) != ENGRAM_OK) { didfmass[j] = 0; continue; }
        for (o = 0; o < eq->n; o++) m += idf_classic(df_of(eq->key[eq->occ[o]])) * eq->qw[eq->occ[o]];
        didfmass[j] = m;
    }
    {
        size_t used = 0; for (j = 0; j <= df_mask; j++) used += df_key[j] != 0u;
        fprintf(stderr, "distinct features in store: %lu (avg doc mass %.1f)\n", (unsigned long)used, avgdl);
    }
    s1p = (double *)engram_array(c.n, sizeof *s1p);
    s1i = (double *)engram_array(c.n, sizeof *s1i);
    srt = (double *)engram_array(c.n, sizeof *srt);
    lab_make_queries(&c, test, nsrc, &Q);
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t];
        double qidf[ENGRAM_ENCQ_MAXF], qmass_idf = 0.0;
        unsigned mode;
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
        for (o = 0; o < eq->n; o++) { qidf[o] = idf_classic(df_of(eq->key[eq->occ[o]])); qmass_idf += qidf[o] * eq->qw[eq->occ[o]]; }
        /* stage 1, both ways, from the same signature bits */
        for (j = 0; j < c.n; j++) {
            const uint64_t *sg = sig + j * ENGRAM_SIG_WORDS;
            double fp = 0.0, fi = 0.0;
            for (o = 0; o < eq->n; o++) {
                unsigned b0 = eq->sbit[o][0], b1 = eq->sbit[o][1];
                if (((sg[b0 >> 6] >> (b0 & 63u)) & 1u) && ((sg[b1 >> 6] >> (b1 & 63u)) & 1u)) {
                    fp += eq->qw[eq->occ[o]]; fi += qidf[o] * eq->qw[eq->occ[o]];
                }
            }
            s1p[j] = fp / eq->qmass; s1i[j] = qmass_idf > 0 ? fi / qmass_idf : 0.0;
        }
        for (mode = 0; mode < 2u; mode++) {
            const double *s1 = mode ? s1i : s1p;
            memcpy(srt, s1, c.n * sizeof *srt);
            qsort(srt, c.n, sizeof *srt, double_desc);
            for (k = 0; k < NC; k++) {
                double thr = srt[(CS[k] < c.n ? CS[k] : c.n) - 1u], src_sc[NS2], sc[NS2];
                unsigned f, beaten[NS2];
                if (s1[i] < thr) continue;
                s1c[mode][k]++;
                memset(beaten, 0, sizeof beaten);
                for (j = 0; j < c.n + 1u; j++) {       /* j == c.n computes the source's own scores first */
                    size_t d = (j == 0u) ? i : j - 1u;
                    double e = 0.0, num_idf = 0.0, cont = 0.0, bm[5];
                    if (j > 0u && (d == i || s1[d] < thr)) continue;
                    engram_encq_score(eq, c.line[d], c.len[d], &e);
                    memset(bm, 0, sizeof bm);
                    for (o = 0; o < eq->n; o++) {
                        size_t sl = eq->occ[o];
                        double qv = eq->qw[sl], dv = eq->dw[sl];
                        if (dv <= 0.0) continue;
                        num_idf += qidf[o] * sqrt(qv * dv);
                        cont += qidf[o] * qv;
                        {
                            static const double K1[5] = { 0.5, 1.2, 1.2, 1.2, 2.0 }, B[5] = { 0.0, 0.0, 0.3, 0.75, 0.3 };
                            unsigned z;
                            for (z = 0; z < 5u; z++) {
                                double norm = K1[z] * (1.0 - B[z] + B[z] * dmass[d] / avgdl);
                                bm[z] += idf_bm25(df_of(eq->key[sl])) * qv * dv * (K1[z] + 1.0) / (dv + norm);
                            }
                        }
                    }
                    sc[0] = e; sc[1] = e;
                    sc[2] = qmass_idf > 0 ? cont / qmass_idf : 0.0;
                    sc[3] = (qmass_idf > 0 && didfmass[d] > 0) ? num_idf / sqrt(qmass_idf * didfmass[d]) : 0.0;
                    sc[4] = bm[0]; sc[5] = bm[1]; sc[6] = bm[2]; sc[7] = bm[3]; sc[8] = bm[4];
                    sc[9] = s1i[d];
                    sc[10] = e * sc[2];
                    sc[11] = bm[2] * e;
                    if (j == 0u) { memcpy(src_sc, sc, sizeof sc); continue; }
                    for (f = 0; f < NS2; f++) if (sc[f] > src_sc[f]) beaten[f] = 1;
                }
                for (f = 0; f < NS2; f++) hit[mode][k][f] += !beaten[f];
            }
        }
    }
    for (k = 0; k < NC; k++) {
        unsigned f, mode;
        for (mode = 0; mode < 2u; mode++) {
            printf("N=%lu %s C=%-3u stage1=%-9s s1_recall %.4f |", (unsigned long)c.n, test ? "test" : "dev",
                   CS[k], mode ? "idf" : "plain", (double)s1c[mode][k] / (double)Q.n);
            for (f = 0; f < NS2; f++) if (!(mode == 0 && f == 1) && !(mode == 1 && f == 0))
                printf(" [%s] %.4f", S2NAME[f], (double)hit[mode][k][f] / (double)Q.n);
            printf("\n");
        }
    }
    return 0;
}
