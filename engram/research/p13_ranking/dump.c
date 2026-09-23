/* solo/rarity4.c -- THE CONTROL for alignment: cues that are NOT contiguous text of the source.
 *   kw3 / kw5 / kw8   3, 5 or 8 distinct pieces of the source (letter runs; ideograph runs cut into
 *                     pairs) in random order, joined by spaces -- a "what was that about" cue
 *   shuf              the words of a 15% fragment, shuffled -- same bag as a fragment, no order
 * Alignment may win on fragments only if it does not lose here.
 * (derived from rarity3.c:)
 * Adds ALIGNMENT: the optimal-string-alignment edit distance of the cue against the best-matching
 * substring of the candidate (Sellers' semi-global DP, transposition = 1 edit), on the normalised
 * codepoint stream the encoder itself hashes (folded, IGNOREs dropped, space runs collapsed).
 * Ties are counted honestly (opt / exp / pess, see rarity2.c).
 * usage: ENGRAM_TEST_DATA=.. ./rarity3 N dev|test nsrc */
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

#define NF 9
static const char *FN[NF] = {
    "bhatt", "lex(cont, bhatt)", "lex(cidf, bhatt)", "lex(-ED, cidf, bhatt)", "lex(cidf, -ED, bhatt)",
    "lex(gated -ED <= |q|/5, cidf, bhatt)", "lex(gated -ED <= |q|/3, cidf, bhatt)",
    "lex(-ED, cont, bhatt)", "lex(gated -ED <= |q|/3, cont, bhatt)"
};
#define NARM4 4
static const char *ARM4[NARM4] = { "kw3", "kw5", "kw8", "shuf" };

/* pieces of a normalised codepoint stream: letter runs; ideograph runs cut into pairs */
static size_t pieces(const uint32_t *cp, size_t n, size_t *ps, size_t *pl, size_t cap)
{
    size_t i = 0, k = 0;
    while (i < n && k < cap) {
        unsigned cl = engram_cp_class(cp[i]);
        size_t s = i;
        if (cl == ENGRAM_CLASS_IDEO) {
            while (i < n && i - s < 2u && engram_cp_class(cp[i]) == ENGRAM_CLASS_IDEO) i++;
            if (i - s == 2u) { ps[k] = s; pl[k] = 2; k++; }
            continue;
        }
        if (cl == ENGRAM_CLASS_WORD) {
            while (i < n && engram_cp_class(cp[i]) == ENGRAM_CLASS_WORD) i++;
            if (i - s >= 3u) { ps[k] = s; pl[k] = i - s; k++; }
            continue;
        }
        i++;
    }
    return k;
}

static void make_control(const engram_enc_cfg *cfg, const corpus *c, int test, size_t nsrc, lab_queries *Q)
{
    static uint32_t cp[MAXN], out[MAXN];
    static size_t ps[MAXN], pl[MAXN], perm[MAXN];
    static char txt[8192];
    size_t t, a;
    Q->n = 0;
    Q->src = (size_t *)engram_array(nsrc * NARM4, sizeof *Q->src);
    Q->len = (size_t *)engram_array(nsrc * NARM4, sizeof *Q->len);
    Q->txt = (char **)engram_array(nsrc * NARM4, sizeof *Q->txt);
    for (t = (size_t)test; t < 2u * nsrc; t += 2u) {
        size_t i = (t * 7919u + 13u) % c->n, n = norm_cps(cfg, c->line[i], c->len[i], cp), np, x, m;
        np = pieces(cp, n, ps, pl, MAXN);
        for (a = 0; a < NARM4; a++) {
            engram_rng r;
            size_t want, lo = 0, hi = np;
            engram_rng_seed(&r, engram_mix2(0xC0DE0u + a, (uint64_t)i));
            if (a == 3u) {                           /* the pieces of a random 15% window, shuffled */
                size_t span = (np * 15u + 99u) / 100u;
                if (span < 2u) span = 2u;
                if (span > np) span = np;
                lo = np > span ? (size_t)engram_rng_below(&r, (uint64_t)(np - span + 1u)) : 0u;
                hi = lo + span; want = span;
            } else want = a == 0u ? 3u : a == 1u ? 5u : 8u;
            for (x = lo; x < hi; x++) perm[x - lo] = x;
            for (x = hi - lo; x > 1u; x--) {        /* Fisher-Yates over the window */
                size_t y = (size_t)engram_rng_below(&r, (uint64_t)x), tmp = perm[x - 1u];
                perm[x - 1u] = perm[y]; perm[y] = tmp;
            }
            if (want > hi - lo) want = hi - lo;
            m = 0;
            for (x = 0; x < want; x++) {
                size_t z;
                if (m) out[m++] = ' ';
                for (z = 0; z < pl[perm[x]]; z++) out[m++] = cp[ps[perm[x]] + z];
            }
            Q->src[Q->n] = i;
            Q->len[Q->n] = encode_all(out, m, txt, sizeof txt);
            Q->txt[Q->n] = (char *)engram_malloc(Q->len[Q->n] + 1u);
            memcpy(Q->txt[Q->n], txt, Q->len[Q->n] + 1u);
            Q->n++;
        }
    }
}

int main(int argc, char **argv)
{
    /* dump.c: per (query, candidate) features for offline rule evaluation.
     * usage: ./dump N dev|test nsrc frag|ctrl  -> dump_<fam>_<split>.tsv
     * columns: t arm cand is_src s1 in50 in200 cont bhatt ed qn dn */
    corpus all, c;
    engram_enc_cfg cfg;
    lab_queries Q;
    size_t N, nsrc, t, j, o;
    int test, ctrl;
    uint64_t *sig;
    engram_encq *eq = (engram_encq *)engram_malloc(sizeof *eq);
    double *s1p, *srt;
    static uint32_t qc[MAXN], dc[MAXN];
    char name[64];
    FILE *out;
    if (argc < 5) return 2;
    N = (size_t)atoll(argv[1]); test = strcmp(argv[2], "test") == 0; nsrc = (size_t)atoll(argv[3]);
    ctrl = strcmp(argv[4], "ctrl") == 0;
    if (!corpus_load(&all, "scale")) return 2;
    c = all; if (N < c.n) c.n = N;
    engram_enc_cfg_default(&cfg);
    sig = sig_build(&cfg, &c);
    s1p = (double *)engram_array(c.n, sizeof *s1p);
    srt = (double *)engram_array(c.n, sizeof *srt);
    if (ctrl) make_control(&cfg, &c, test, nsrc, &Q); else lab_make_queries(&c, test, nsrc, &Q);
    snprintf(name, sizeof name, "dump_%s_%s.tsv", ctrl ? "ctrl" : "frag", test ? "test" : "dev");
    out = fopen(name, "w");
    for (t = 0; t < Q.n; t++) {
        size_t i = Q.src[t], qn;
        double thr50, thr200;
        unsigned arms = ctrl ? NARM4 : 12u;
        if (engram_encq_build(eq, &cfg, Q.txt[t], Q.len[t]) != ENGRAM_OK) { fprintf(out, "%lu\t%lu\tFAIL\n", (unsigned long)t, (unsigned long)(t % arms)); continue; }
        qn = norm_cps(&cfg, Q.txt[t], Q.len[t], qc);
        for (j = 0; j < c.n; j++) engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &s1p[j]);
        memcpy(srt, s1p, c.n * sizeof *srt);
        qsort(srt, c.n, sizeof *srt, double_desc);
        thr50 = srt[(50u < c.n ? 50u : c.n) - 1u];
        thr200 = srt[(200u < c.n ? 200u : c.n) - 1u];
        for (j = 0; j < c.n; j++) {
            double e = 0.0, cont = 0.0;
            size_t dn;
            unsigned ed;
            if (!(s1p[j] >= thr200 || j == i)) continue;
            engram_encq_score(eq, c.line[j], c.len[j], &e);
            for (o = 0; o < eq->n; o++) { size_t sl = eq->occ[o]; if (eq->dw[sl] > 0.0) cont += eq->qw[sl]; }
            cont /= eq->qmass;
            dn = norm_cps(&cfg, c.line[j], c.len[j], dc);
            ed = sellers(qc, qn, dc, dn);
            fprintf(out, "%lu\t%lu\t%lu\t%d\t%.17g\t%d\t%d\t%.17g\t%.17g\t%u\t%lu\t%lu\n", (unsigned long)t,
                    (unsigned long)(t % arms), (unsigned long)j, j == i, s1p[j], s1p[j] >= thr50, s1p[j] >= thr200,
                    cont, e, ed, (unsigned long)qn, (unsigned long)dn);
        }
    }
    fclose(out);
    (void)df_of; (void)idf_nat; (void)FN; (void)ARM4;
    return 0;
}
