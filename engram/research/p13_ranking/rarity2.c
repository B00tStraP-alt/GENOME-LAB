/* solo/rarity2.c -- the finalists of rarity.c, measured HONESTLY about ties, and the df question.
 *
 * rarity.c counted the source as first unless some candidate scored STRICTLY higher. That flatters any
 * score with ties -- and containment ties by construction: every candidate holding all of the query's
 * features scores exactly 1. Here every function reports three numbers:
 *   opt   no candidate strictly above the source (ties won)       -- what rarity.c reported
 *   pess  no candidate at or above the source (ties lost)         -- a floor
 *   exp   1 / (1 + ties) when nothing is strictly above           -- a uniformly random tie-break
 * A score is only as good as its PESSIMISTIC or EXPECTED number, unless it carries a tie-break.
 *
 * And where df comes from:
 *   exact      a hash table of every feature of every chunk (2.2 M entries at 21.8 K chunks)
 *   floorT     exact df, but max(df, T): how much does resolving RARE features matter?
 *   sigraw     df counted during the stage-1 scan: docs whose signature holds both bits of the feature
 *              -- no table, nothing to maintain, always the live store; carries Bloom false positives
 *   sigcorr    sigraw minus the expected false positives, sum_j fill_j^2, clamped at 0
 *   fixed      exact df, idf by integer log2 (Q16) -- the R5-deterministic form
 * usage: ENGRAM_TEST_DATA=. ./rarity2 N dev|test nsrc */
#define LAB_NO_MAIN
#include "lab.c"
#include <math.h>

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

/* floor(65536 * log2(x)) for x >= 1, integers only: exponent from the top bit, then 16 fraction bits by
 * repeated squaring of the Q31 mantissa. Exact and identical on every platform. */
static uint32_t ilog2_q16(uint64_t x)
{
    unsigned e = 63u, i;
    uint64_t m;
    uint32_t r;
    while (!((x >> e) & 1u)) e--;
    m = e >= 31u ? x >> (e - 31u) : x << (31u - e);            /* m in [2^31, 2^32): 1.xxx in Q31 */
    r = (uint32_t)e << 16;
    for (i = 0; i < 16u; i++) {
        m = (m * m) >> 31;                                      /* m^2 in [2^31, 2^33) */
        if (m >= ((uint64_t)1 << 32)) { r |= 1u << (15u - i); m >>= 1; }
    }
    return r;
}

static double NDOC;
static double idf_nat(double df) { return log((NDOC + 1.0) / (df + 0.5)); }
/* log2((N+1)/(df+0.5)) = log2(2N+2) - log2(2df+1) */
static double idf_fixed(uint32_t df)
{
    return (double)(ilog2_q16(2u * (uint64_t)NDOC + 2u) - ilog2_q16(2u * (uint64_t)df + 1u)) / 65536.0;
}

enum { DF_EXACT, DF_F4, DF_F16, DF_F64, DF_F256, DF_SIGRAW, DF_SIGCORR, DF_FIXED, NDF };
static const char *DFNAME[NDF] = { "exact", "floor4", "floor16", "floor64", "floor256", "sigraw", "sigcorr", "fixed" };

/* scoring functions: 0 bhatt; 1 cont (no idf); 2..2+NDF-1 cont-idf per df source; then
 * contmin-idf(exact), lex(cont-idf exact, bhatt), lex(cont-idf sigcorr, bhatt), bm25 1.2/0.75,
 * lex(cont-idf fixed, bhatt) */
#define F_BHATT 0
#define F_CONT 1
#define F_CIDF 2
#define F_CMIN (F_CIDF + NDF)
#define F_LEXE (F_CMIN + 1)
#define F_LEXS (F_CMIN + 2)
#define F_BM25 (F_CMIN + 3)
#define F_LEXF (F_CMIN + 4)
#define NF (F_CMIN + 5)

typedef struct { double a, b; } sc2;             /* primary, secondary (secondary 0 unless lexicographic) */
static int sc_gt(sc2 x, sc2 y) { return x.a > y.a || (x.a == y.a && x.b > y.b); }
static int sc_eq(sc2 x, sc2 y) { return x.a == y.a && x.b == y.b; }

static const char *fname(unsigned f, char *buf)
{
    if (f == F_BHATT) return "bhatt";
    if (f == F_CONT) return "cont (no idf)";
    if (f < F_CMIN) { sprintf(buf, "cont-idf %s", DFNAME[f - F_CIDF]); return buf; }
    if (f == F_CMIN) return "contmin-idf exact";
    if (f == F_LEXE) return "lex(cont-idf exact, bhatt)";
    if (f == F_LEXS) return "lex(cont-idf sigcorr, bhatt)";
    if (f == F_BM25) return "bm25 1.2/0.75";
    return "lex(cont-idf fixed, bhatt)";
}

int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j, o, ncand;
    int test;
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    double *s1p, *srt, *dmass, avgdl = 0.0, efp = 0.0, maxdiff = 0.0;
    size_t *cand;
    sc2 *csc;                                     /* per candidate x function */
    unsigned k, s1c[NC];
    double opt[NC][NF], pess[NC][NF], expv[NC][NF];
    memset(s1c, 0, sizeof s1c); memset(opt, 0, sizeof opt); memset(pess, 0, sizeof pess); memset(expv, 0, sizeof expv);
    if (argc < 4) return 2;
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    NDOC = (double)c.n;
    engram_enc_cfg_default(&cfg);
    sig = sig_build(&cfg, &c);
    for (j = 0; j < c.n; j++) {                   /* expected false-positive count of a random feature */
        unsigned w, pop = 0;
        for (w = 0; w < ENGRAM_SIG_WORDS; w++) {
            uint64_t v = sig[j * ENGRAM_SIG_WORDS + w];
            while (v) { v &= v - 1u; pop++; }
        }
        efp += ((double)pop / ENGRAM_SIG_BITS) * ((double)pop / ENGRAM_SIG_BITS);
    }
    df_mask = (1u << 24) - 1u;
    df_key = (uint64_t *)calloc(df_mask + 1u, sizeof *df_key);
    df_val = (uint32_t *)calloc(df_mask + 1u, sizeof *df_val);
    dmass = (double *)engram_array(c.n, sizeof *dmass);
    for (j = 0; j < c.n; j++) {
        if (engram_encq_build(eq, &cfg, c.line[j], c.len[j]) != ENGRAM_OK) { dmass[j] = 0; continue; }
        dmass[j] = eq->qmass; avgdl += eq->qmass;
        for (o = 0; o < eq->n; o++) (*df_slot(eq->key[eq->occ[o]], 1))++;
    }
    avgdl /= (double)c.n;
    {
        uint32_t d;
        for (d = 0; d <= (uint32_t)NDOC; d++) {
            double a = idf_fixed(d) * log(2.0), b = idf_nat((double)d), e = fabs(a - b);
            if (e > maxdiff) maxdiff = e;
        }
    }
    fprintf(stderr, "N %lu  expected FP df per feature %.1f  fixed-vs-float idf max |diff| %.2e nats\n",
            (unsigned long)c.n, efp, maxdiff);
    s1p = (double *)engram_array(c.n, sizeof *s1p);
    srt = (double *)engram_array(c.n, sizeof *srt);
    cand = (size_t *)engram_array(c.n, sizeof *cand);
    csc = (sc2 *)engram_array(c.n * NF, sizeof *csc);
    lab_make_queries(&c, test, nsrc, &Q);
    FILE *qd = fopen(test ? "rarity2_q_test.tsv" : "rarity2_q_dev.tsv", "w");
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t], srcpos = (size_t)-1;
        double idf[NDF][ENGRAM_ENCQ_MAXF], den[NDF], thr[NC];
        uint32_t dsig[ENGRAM_ENCQ_MAXF];
        unsigned d;
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
        memset(dsig, 0, sizeof dsig);
        for (j = 0; j < c.n; j++) {               /* stage 1 (plain) + signature df, in one scan */
            const uint64_t *sg = sig + j * ENGRAM_SIG_WORDS;
            double fp = 0.0;
            for (o = 0; o < eq->n; o++) {
                unsigned b0 = eq->sbit[o][0], b1 = eq->sbit[o][1];
                if (((sg[b0 >> 6] >> (b0 & 63u)) & 1u) && ((sg[b1 >> 6] >> (b1 & 63u)) & 1u)) {
                    fp += eq->qw[eq->occ[o]]; dsig[o]++;
                }
            }
            s1p[j] = fp / eq->qmass;
        }
        for (o = 0; o < eq->n; o++) {
            uint32_t df = df_of(eq->key[eq->occ[o]]);
            double corr = (double)dsig[o] - efp;
            idf[DF_EXACT][o] = idf_nat((double)df);
            idf[DF_F4][o] = idf_nat((double)(df > 4u ? df : 4u));
            idf[DF_F16][o] = idf_nat((double)(df > 16u ? df : 16u));
            idf[DF_F64][o] = idf_nat((double)(df > 64u ? df : 64u));
            idf[DF_F256][o] = idf_nat((double)(df > 256u ? df : 256u));
            idf[DF_SIGRAW][o] = idf_nat((double)dsig[o]);
            idf[DF_SIGCORR][o] = idf_nat(corr > 0.0 ? corr : 0.0);
            idf[DF_FIXED][o] = idf_fixed(df);
        }
        for (d = 0; d < NDF; d++) {
            den[d] = 0.0;
            for (o = 0; o < eq->n; o++) den[d] += idf[d][o] * eq->qw[eq->occ[o]];
        }
        memcpy(srt, s1p, c.n * sizeof *srt);
        qsort(srt, c.n, sizeof *srt, double_desc);
        for (k = 0; k < NC; k++) thr[k] = srt[(CS[k] < c.n ? CS[k] : c.n) - 1u];
        /* candidates: everything at or above the widest threshold; the source is scored even when out */
        ncand = 0;
        for (j = 0; j < c.n; j++) if (s1p[j] >= thr[NC - 1] || j == i) cand[ncand++] = j;
        for (j = 0; j < ncand; j++) {
            size_t dd = cand[j];
            double e = 0.0, cont = 0.0, cidf[NDF], cmin = 0.0, bm = 0.0;
            sc2 *r = csc + j * NF;
            unsigned f;
            if (dd == i) srcpos = j;
            engram_encq_score(eq, c.line[dd], c.len[dd], &e);
            memset(cidf, 0, sizeof cidf);
            for (o = 0; o < eq->n; o++) {
                size_t sl = eq->occ[o];
                double qv = eq->qw[sl], dv = eq->dw[sl];
                if (dv <= 0.0) continue;
                cont += qv;
                for (d = 0; d < NDF; d++) cidf[d] += idf[d][o] * qv;
                cmin += idf[DF_EXACT][o] * (qv < dv ? qv : dv);
                {
                    double norm = 1.2 * (1.0 - 0.75 + 0.75 * dmass[dd] / avgdl);
                    double df = (double)df_of(eq->key[sl]);
                    bm += log(1.0 + (NDOC - df + 0.5) / (df + 0.5)) * qv * dv * 2.2 / (dv + norm);
                }
            }
            for (f = 0; f < NF; f++) r[f].b = 0.0;
            r[F_BHATT].a = e;
            r[F_CONT].a = cont / eq->qmass;
            for (d = 0; d < NDF; d++) r[F_CIDF + d].a = den[d] > 0.0 ? cidf[d] / den[d] : 0.0;
            r[F_CMIN].a = den[DF_EXACT] > 0.0 ? cmin / den[DF_EXACT] : 0.0;
            r[F_LEXE] = r[F_CIDF + DF_EXACT]; r[F_LEXE].b = e;
            r[F_LEXS] = r[F_CIDF + DF_SIGCORR]; r[F_LEXS].b = e;
            r[F_BM25].a = bm;
            r[F_LEXF] = r[F_CIDF + DF_FIXED]; r[F_LEXF].b = e;
        }
        for (k = 0; k < NC; k++) {
            unsigned f;
            if (s1p[i] < thr[k]) { if (k == 1u) fprintf(qd, "%lu\t%lu\tOUT\n", (unsigned long)t, (unsigned long)i); continue; }
            if (k == 1u) fprintf(qd, "%lu\t%lu\t", (unsigned long)t, (unsigned long)i);
            s1c[k]++;
            for (f = 0; f < NF; f++) {
                sc2 s = csc[srcpos * NF + f];
                unsigned above = 0, ties = 0;
                for (j = 0; j < ncand; j++) {
                    sc2 x = csc[j * NF + f];
                    if (j == srcpos || s1p[cand[j]] < thr[k]) continue;
                    if (sc_gt(x, s)) above++; else if (sc_eq(x, s)) ties++;
                }
                if (!above) { opt[k][f] += 1.0; expv[k][f] += 1.0 / (1.0 + ties); if (!ties) pess[k][f] += 1.0; }
                if (k == 1u && (f == F_BHATT || f == F_CONT || f == F_CIDF || f == F_LEXE))
                    fprintf(qd, "%s%u %u", f == F_BHATT ? "" : "\t", above, ties);
            }
            if (k == 1u) fprintf(qd, "\n");
        }
    }
    fclose(qd);
    {   /* the queries themselves, for the ambiguity analysis */
        FILE *qt = fopen(test ? "queries_test.tsv" : "queries_dev.tsv", "w");
        for (t = 0; t < Q.n; t++) { fprintf(qt, "%lu\t%lu\t", (unsigned long)t, (unsigned long)Q.src[t]); fwrite(Q.txt[t], 1, Q.len[t], qt); fputc('\n', qt); }
        fclose(qt);
    }
    for (k = 0; k < NC; k++) {
        unsigned f;
        char buf[64];
        printf("N=%lu %s C=%u s1_recall %.4f\n", (unsigned long)c.n, test ? "test" : "dev", CS[k],
               (double)s1c[k] / (double)Q.n);
        for (f = 0; f < NF; f++)
            printf("  %-30s opt %.4f  exp %.4f  pess %.4f\n", fname(f, buf), opt[k][f] / Q.n, expv[k][f] / Q.n,
                   pess[k][f] / Q.n);
    }
    return 0;
}
