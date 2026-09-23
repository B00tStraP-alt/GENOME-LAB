/* solo/rarity3.c -- is there headroom past lex(containment-idf, bhatt), and where is it?
 * Adds ALIGNMENT: the optimal-string-alignment edit distance of the cue against the best-matching
 * substring of the candidate (Sellers' semi-global DP, transposition = 1 edit), on the normalised
 * codepoint stream the encoder itself hashes (folded, IGNOREs dropped, space runs collapsed).
 * Ties are counted honestly (opt / exp / pess, see rarity2.c).
 * usage: ENGRAM_TEST_DATA=. ./rarity3 N dev|test nsrc */
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
        if (df_key[i] == 0u) { if (!insert) return NULL; df_key[i] = k; df_val[i] = 0; return &df_val[i]; }
        i = (i + 1u) & df_mask;
    }
}
static uint32_t df_of(uint64_t k) { uint32_t *v = df_slot(k, 0); return v ? *v : 0u; }
static double NDOC;
static double idf_nat(double df) { return log((NDOC + 1.0) / (df + 0.5)); }

#define MAXN 2048
static size_t norm_cps(const engram_enc_cfg *cfg, const char *s, size_t n, uint32_t *out)
{
    engram_textit it;
    uint32_t cp;
    unsigned cls;
    size_t k = 0;
    int sp = 0;
    if (engram_textit_init(&it, s, n, cfg->fold, ENGRAM_UTF8_REPLACE) != ENGRAM_OK) return 0;
    while (k < MAXN && engram_textit_next(&it, &cp, &cls)) {
        if (cls == ENGRAM_CLASS_SPACE) { sp = k > 0; continue; }
        if (sp && k < MAXN) { out[k++] = ' '; sp = 0; }
        if (k < MAXN) out[k++] = cp;
    }
    return k;
}

/* min over end positions of the OSA distance of q[0..m) against a substring of d[0..n) */
static unsigned sellers(const uint32_t *q, size_t m, const uint32_t *d, size_t n)
{
    static unsigned R[3][MAXN + 1];
    unsigned *pp = R[0], *p = R[1], *c = R[2], best;
    size_t i, j;
    /* rows indexed by query position i, columns by doc position j; row 0 is all zeros (free start) */
    for (j = 0; j <= n; j++) p[j] = 0;
    for (j = 0; j <= n; j++) pp[j] = 0;
    for (i = 1; i <= m; i++) {
        c[0] = (unsigned)i;
        for (j = 1; j <= n; j++) {
            unsigned v = p[j - 1] + (q[i - 1] != d[j - 1]);
            if (p[j] + 1u < v) v = p[j] + 1u;
            if (c[j - 1] + 1u < v) v = c[j - 1] + 1u;
            if (i > 1 && j > 1 && q[i - 1] == d[j - 2] && q[i - 2] == d[j - 1] && pp[j - 2] + 1u < v) v = pp[j - 2] + 1u;
            c[j] = v;
        }
        { unsigned *t = pp; pp = p; p = c; c = t; }
    }
    best = (unsigned)m;
    for (j = 0; j <= n; j++) if (p[j] < best) best = p[j];
    return best;
}

typedef struct { double a, b, c; } sc3;
static int sc_gt(sc3 x, sc3 y) { return x.a != y.a ? x.a > y.a : x.b != y.b ? x.b > y.b : x.c > y.c; }
static int sc_eq(sc3 x, sc3 y) { return x.a == y.a && x.b == y.b && x.c == y.c; }

#define NF 7
static const char *FN[NF] = {
    "bhatt", "lex(cont, bhatt)", "lex(cidf, bhatt)", "lex(-ED, cidf, bhatt)", "lex(cidf, -ED, bhatt)",
    "lex(gated -ED <= |q|/5, cidf, bhatt)", "lex(gated -ED <= |q|/3, cidf, bhatt)"
};

int main(int argc, char **argv)
{
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j, o, ncand;
    int test;
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    double *s1p, *srt, t_aln = 0.0;
    size_t *cand, n_aln = 0;
    sc3 *csc;
    unsigned k, s1c[NC];
    static uint32_t qc[MAXN], dc[MAXN];
    double opt[NC][NF], pess[NC][NF], expv[NC][NF];
    unsigned miss[NF][12];
    FILE *qd;
    memset(s1c, 0, sizeof s1c); memset(opt, 0, sizeof opt); memset(pess, 0, sizeof pess); memset(expv, 0, sizeof expv);
    memset(miss, 0, sizeof miss);
    if (argc < 4) return 2;
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    NDOC = (double)c.n;
    engram_enc_cfg_default(&cfg);
    sig = sig_build(&cfg, &c);
    df_mask = (1u << 24) - 1u;
    df_key = (uint64_t *)calloc(df_mask + 1u, sizeof *df_key);
    df_val = (uint32_t *)calloc(df_mask + 1u, sizeof *df_val);
    for (j = 0; j < c.n; j++) {
        if (engram_encq_build(eq, &cfg, c.line[j], c.len[j]) != ENGRAM_OK) continue;
        for (o = 0; o < eq->n; o++) (*df_slot(eq->key[eq->occ[o]], 1))++;
    }
    s1p = (double *)engram_array(c.n, sizeof *s1p);
    srt = (double *)engram_array(c.n, sizeof *srt);
    cand = (size_t *)engram_array(c.n, sizeof *cand);
    csc = (sc3 *)engram_array(c.n * NF, sizeof *csc);
    lab_make_queries(&c, test, nsrc, &Q);
    qd = fopen(test ? "rarity3_q_test.tsv" : "rarity3_q_dev.tsv", "w");
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t], srcpos = (size_t)-1, qn;
        double idf[ENGRAM_ENCQ_MAXF], den = 0.0, thr[NC];
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) continue;
        qn = norm_cps(&cfg, Q.txt[t], Q.len[t], qc);
        for (j = 0; j < c.n; j++) engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &s1p[j]);
        for (o = 0; o < eq->n; o++) { idf[o] = idf_nat((double)df_of(eq->key[eq->occ[o]])); den += idf[o] * eq->qw[eq->occ[o]]; }
        memcpy(srt, s1p, c.n * sizeof *srt);
        qsort(srt, c.n, sizeof *srt, double_desc);
        for (k = 0; k < NC; k++) thr[k] = srt[(CS[k] < c.n ? CS[k] : c.n) - 1u];
        ncand = 0;
        for (j = 0; j < c.n; j++) if (s1p[j] >= thr[NC - 1] || j == i) cand[ncand++] = j;
        for (j = 0; j < ncand; j++) {
            size_t dd = cand[j], dn;
            double e = 0.0, cont = 0.0, cidf = 0.0, t0;
            unsigned ed;
            sc3 *r = csc + j * NF;
            if (dd == i) srcpos = j;
            engram_encq_score(eq, c.line[dd], c.len[dd], &e);
            for (o = 0; o < eq->n; o++) {
                size_t sl = eq->occ[o];
                if (eq->dw[sl] <= 0.0) continue;
                cont += eq->qw[sl]; cidf += idf[o] * eq->qw[sl];
            }
            cont /= eq->qmass; cidf = den > 0.0 ? cidf / den : 0.0;
            t0 = (double)engram_now_ns();
            dn = norm_cps(&cfg, c.line[dd], c.len[dd], dc);
            ed = sellers(qc, qn, dc, dn);
            t_aln += (double)engram_now_ns() - t0; n_aln++;
            r[0].a = e; r[0].b = 0; r[0].c = 0;
            r[1].a = cont; r[1].b = e; r[1].c = 0;
            r[2].a = cidf; r[2].b = e; r[2].c = 0;
            r[3].a = -(double)ed; r[3].b = cidf; r[3].c = e;
            r[4].a = cidf; r[4].b = -(double)ed; r[4].c = e;
            r[5].a = ed * 5u <= qn ? -(double)ed : -1e9; r[5].b = cidf; r[5].c = e;
            r[6].a = ed * 3u <= qn ? -(double)ed : -1e9; r[6].b = cidf; r[6].c = e;
        }
        for (k = 0; k < NC; k++) {
            unsigned f;
            if (s1p[i] < thr[k]) {
                if (k == 1u) for (f = 0; f < NF; f++) miss[f][t % 12u]++;
                continue;
            }
            s1c[k]++;
            for (f = 0; f < NF; f++) {
                sc3 s = csc[srcpos * NF + f];
                unsigned above = 0, ties = 0;
                for (j = 0; j < ncand; j++) {
                    sc3 x = csc[j * NF + f];
                    if (j == srcpos || s1p[cand[j]] < thr[k]) continue;
                    if (sc_gt(x, s)) above++; else if (sc_eq(x, s)) ties++;
                }
                if (!above) { opt[k][f] += 1.0; expv[k][f] += 1.0 / (1.0 + ties); if (!ties) pess[k][f] += 1.0; }
                if (k == 1u && (above || ties)) miss[f][t % 12u]++;
                if (k == 1u) fprintf(qd, "%s%u %u", f ? "\t" : "", above, ties);
            }
            if (k == 1u) fprintf(qd, "\n");
        }
    }
    fclose(qd);
    fprintf(stderr, "alignment: %.1f us per candidate\n", t_aln / (double)n_aln / 1e3);
    for (k = 0; k < NC; k++) {
        unsigned f;
        printf("N=%lu %s C=%u s1_recall %.4f\n", (unsigned long)c.n, test ? "test" : "dev", CS[k], (double)s1c[k] / (double)Q.n);
        for (f = 0; f < NF; f++)
            printf("  %-38s opt %.4f  exp %.4f  pess %.4f\n", FN[f], opt[k][f] / Q.n, expv[k][f] / Q.n, pess[k][f] / Q.n);
    }
    {
        unsigned f, a;
        static const unsigned P[4] = { 5, 8, 10, 15 };
        printf("C=50 misses (incl. ties) by arm pct/typos:\n");
        for (f = 0; f < NF; f++) {
            printf("  %-38s", FN[f]);
            for (a = 0; a < 12u; a++) printf(" %u/%u:%-3u", P[a % 4u], a / 4u, miss[f][a]);
            printf("\n");
        }
    }
    return 0;
}
