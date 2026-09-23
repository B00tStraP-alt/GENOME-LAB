/* ==================================================================================================
 * test_enc.c -- P1.2 (part 2): the encoder, proven and MEASURED.
 * ==================================================================================================
 *
 * THE CLAIMS, AND WHERE EACH THRESHOLD CAME FROM
 * ==================================================================================================
 * Commandment: a threshold is written before the measurement it judges, never tuned after it. Where
 * a measurement forced a claim to be RESTATED, the old statement, the reason and the new one are all
 * recorded here and in the COMMANDMENTS.md ledger -- a restatement is not allowed to be silent.
 *
 * STATED BEFORE ANY MEASUREMENT, UNCHANGED (from VECTRA TRACE's reported behaviour):
 *   T1  every encoded vector is a unit vector                           |norm - 1| < 1e-5
 *   T2  encoding is deterministic, and ALLOCATES NOTHING                bit-identical; 0 allocations
 *   T3  whole paragraph, 1 typo      recall@1 >= 0.98
 *   T4  whole paragraph, 3 typos     recall@1 >= 0.95
 *   T5  whole paragraph, 5 typos     recall@1 >= 0.90
 *   T6  whole paragraph, 3 swaps of adjacent words              recall@1 >= 0.95
 *   T9  FOLDING: under COMPAT a paragraph and its own normalised form encode BIT-IDENTICALLY, and a
 *       CASE-only control loses source similarity on the same unaccented queries.
 *   T10 SEPARATION: a 3-typo query's cosine to its source exceeds the mean unrelated-pair cosine by
 *       more than three standard deviations of the unrelated distribution.
 *   T11 Latin-script languages (fr, de, sv), whole paragraph, 3 typos: recall@1 >= 0.90.
 *
 * RESTATED -- each with the ledger row that records what failed and why:
 *   W-P1.2-6   T7 and T8 were first stated on whole paragraphs with 5 typos. Every arm -- every
 *              ablation, every dimension -- scored recall@1 = 1.0000 there, so neither gate could ever
 *              have rejected anything. Both now run on the FRAGMENT OBJECTIVE below.
 *   W-P1.2-10  T8's first cascade put the dense vector in front of the exact stage, and T8 failed
 *              it: exhaustive exact search beat it beyond the margin. The shipped first stage is now
 *              the signature; the dense-first cascade stays in the suite as the control it lost to.
 *   W-P1.2-12  T7 and T12 then failed on DENSE recall, and the defaults changed because of it
 *              (3-grams 1 -> 1/16). T7 now judges the cascade users get; T12, which is about the
 *              dense vector's own shaping, judges the dense stage. Every dense-only difference in
 *              T7 is still printed, marked "reported, not gated".
 *
 *   T7  ABLATION: no single-class change -- removing a class that is on, restoring a former weight,
 *       or turning on a class that is off at its former VECTRA TRACE weight -- improves the CASCADE
 *       beyond the paired margin. Removing the character classes altogether must LOSE beyond it:
 *       the gate is shown able to see a difference in the direction it is guarding.
 *   T8  RETRIEVAL AT THE SHIPPED GEOMETRY: the cascade (signature containment over everything, the
 *       top 50 re-scored exactly) beats dense-only 512 beyond the margin; neither a dense first
 *       stage nor an EXHAUSTIVE exact search beats it beyond the margin; and no dense-only
 *       dimension up to 4096 does either.
 *
 * ADDED, FROM THE MEASUREMENTS THAT CHANGED THE DESIGN (engram_enc.h has the numbers):
 *   T12 TF SHAPING of the dense stage: signed square root beats linear beyond the margin;
 *       quarter-power and sign do not beat square root beyond it.
 *   T13 SCRIPT SPLIT: on Chinese, removing the ideographic features loses beyond the margin, and no
 *       change of their weight (4, 64), removal of the character classes, or ordered bigrams wins
 *       beyond it (judged on the dense stage: the Chinese cascade is saturated). On every one of the
 *       five languages the default beats the inherited VECTRA TRACE weighting beyond the margin, on
 *       held-out paragraphs.
 *   T14 THE EXACT STAGE equals an independently computed Bhattacharyya coefficient to 1e-12, is
 *       symmetric, is 1 for identical texts, and allocates nothing.
 *   T15 THE SIGNATURE has no false negatives -- every text scores exactly 1 against its own
 *       signature -- sets at most K bits per feature, leaves no bits behind on refusal, and
 *       allocates nothing.
 *   F   REGRESSION FLOORS. The fragment figures are MEASURED values, not advance claims; each floor
 *       is the measurement minus two clustered standard errors, recorded with its provenance, so
 *       that a later change cannot silently give back what was measured.
 *
 * THE FRAGMENT OBJECTIVE -- THE TASK THAT DECIDES
 * ==================================================================================================
 * A person half-remembers a PIECE of what they wrote. Twelve arms: 5, 8, 10 and 15 percent of a
 * paragraph, starting at a word boundary, with 0, 1 and 2 typos. The index holds every paragraph; the
 * queries come from the HELD-OUT odd paragraphs only, because the even ones were the development
 * split on which the weights were chosen. The score is the mean recall@1 over all twelve arms.
 *
 * THE PAIRED MARGIN. Two configurations answer the SAME queries, so they are compared query by query,
 * not as two independent proportions. The twelve arms cut from one paragraph are not independent of
 * each other, so the unit is the paragraph: per paragraph, the mean over its arms of (hit_B - hit_A);
 * the margin is two standard errors of the mean of those, plus one query. A difference inside it is a
 * tie, and no gate decides on a tie -- the defect that made VECTRA TRACE's default flip twice on
 * fourth-decimal noise.
 *
 * SATURATION GUARD. If the default's objective reaches 0.98 the instrument can no longer see an
 * improvement, and the suite FAILS rather than passing gates that cannot fail (rule R6).
 *
 * CROSS-PLATFORM IDENTITY. "FINGERPRINT <hex>" hashes every float of every dense vector of every
 * corpus, "SIGPRINT <hex>" every signature, "EXACTPRINT <hex>" a matrix of exact scores.
 * tools/gate.sh requires the Linux (gcc) and Windows (mingw, under Wine) values to be equal -- rule
 * R5, demonstrated rather than asserted.
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_enc.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "../src/engram_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXCP     4096u
#define MAXB      16384u
#define NARM      12u
#define CASCADE_C 50u
#define SATURATED 0.98

static const unsigned ARM_PCT[4] = { 5u, 8u, 10u, 15u };

static int g_quick;

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

/* ==================================================================================================
 * CORPUS
 * ============================================================================================== */
typedef struct {
    uint8_t     *buf;
    const char **line;
    size_t      *len;
    size_t       n;
} corpus;

static int corpus_load(corpus *c, const char *lang)
{
    char name[64], *path;
    size_t blen = 0, i, cap = 0;
    memset(c, 0, sizeof *c);
    snprintf(name, sizeof name, "corpus_%s.txt", lang);
    path = engram_path_join(data_dir(), name);
    if (!path) return 0;
    if (engram_file_read(path, &c->buf, &blen) != ENGRAM_OK) { engram_free(path); return 0; }
    engram_free(path);
    for (i = 0; i < blen; ) {
        size_t s = i;
        while (i < blen && c->buf[i] != '\n') i++;
        if (i > s) {
            if (engram_grow((void **)&c->line, &cap, c->n + 1u, sizeof *c->line) != ENGRAM_OK) return 0;
            c->n++;
        }
        i++;
    }
    c->len = (size_t *)engram_array(c->n ? c->n : 1u, sizeof *c->len);
    if (!c->len) return 0;
    {
        size_t k = 0;
        for (i = 0; i < blen && k < c->n; ) {
            size_t s = i;
            while (i < blen && c->buf[i] != '\n') i++;
            if (i > s) { c->line[k] = (const char *)c->buf + s; c->len[k] = i - s; k++; }
            i++;
        }
    }
    return 1;
}

static void corpus_free(corpus *c)
{
    engram_free(c->buf);
    engram_free((void *)c->line);
    engram_free(c->len);
    memset(c, 0, sizeof *c);
}

/* ==================================================================================================
 * CODEPOINT HELPERS (the corpora are valid UTF-8; test_text proves the decoder)
 * ============================================================================================== */
static size_t decode_all(const char *s, size_t n, uint32_t *out, size_t cap)
{
    size_t off = 0, k = 0;
    while (off < n && k < cap) {
        uint32_t cp;
        int valid;
        off += engram_utf8_decode((const uint8_t *)s + off, n - off, &cp, &valid);
        out[k++] = cp;
    }
    return k;
}

static size_t encode_all(const uint32_t *cp, size_t n, char *out, size_t cap)
{
    size_t i, k = 0;
    for (i = 0; i < n; i++) {
        uint8_t e[4];
        size_t m = engram_utf8_encode(cp[i], e);
        if (k + m >= cap) break;
        memcpy(out + k, e, m);
        k += m;
    }
    out[k] = 0;
    return k;
}

static int is_letter(uint32_t cp)
{
    unsigned k = engram_cp_class(cp);
    return k == ENGRAM_CLASS_WORD || k == ENGRAM_CLASS_IDEO;
}

typedef struct { uint32_t *a; size_t n; } alphabet;

/* The WORD-class codepoints a corpus actually uses: a substituted or inserted character is drawn from
 * the language's own alphabet, so a French typo is a plausible French character and a Chinese one a
 * real Chinese character, not an arbitrary codepoint no typist could produce. */
static int alphabet_build(alphabet *ab, const corpus *c)
{
    static uint8_t seen[0x110000u / 8u + 1u];
    static uint32_t cps[MAXCP];
    size_t i, j, cap = 0;
    memset(seen, 0, sizeof seen);
    memset(ab, 0, sizeof *ab);
    for (i = 0; i < c->n; i++) {
        size_t m = decode_all(c->line[i], c->len[i], cps, MAXCP);
        for (j = 0; j < m; j++) {
            uint32_t cp = cps[j];
            if (!is_letter(cp) || cp > 0x10FFFFu || (seen[cp >> 3] & (1u << (cp & 7u)))) continue;
            seen[cp >> 3] = (uint8_t)(seen[cp >> 3] | (1u << (cp & 7u)));
            if (engram_grow((void **)&ab->a, &cap, ab->n + 1u, sizeof *ab->a) != ENGRAM_OK) return 0;
            ab->a[ab->n++] = cp;
        }
    }
    return ab->n > 0u;
}

/* ==================================================================================================
 * VARIANT GENERATORS -- the mistakes people actually make
 * ============================================================================================== */
typedef size_t (*variant_fn)(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                             const void *arg);

typedef struct { unsigned k; const alphabet *ab; } typo_arg;

/* k Damerau edits at random LETTER positions: substitute, delete, insert, transpose. */
static size_t v_typos(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                      const void *argp)
{
    const typo_arg *a = (const typo_arg *)argp;
    size_t m = n < cap ? n : cap, t, tries;
    memcpy(out, in, m * sizeof *out);
    for (t = 0; t < a->k; t++) {
        size_t pos = 0;
        unsigned op;
        for (tries = 0; tries < 64u; tries++) {
            pos = (size_t)engram_rng_below(r, (uint64_t)m);
            if (m && is_letter(out[pos])) break;
        }
        if (!m || !is_letter(out[pos])) break;
        op = (unsigned)engram_rng_below(r, 4u);
        if (op == 0u) {                                   /* substitute */
            out[pos] = a->ab->a[engram_rng_below(r, (uint64_t)a->ab->n)];
        } else if (op == 1u && m > 1u) {                  /* delete */
            memmove(out + pos, out + pos + 1, (m - pos - 1u) * sizeof *out);
            m--;
        } else if (op == 2u && m + 1u < cap) {            /* insert */
            memmove(out + pos + 1, out + pos, (m - pos) * sizeof *out);
            out[pos] = a->ab->a[engram_rng_below(r, (uint64_t)a->ab->n)];
            m++;
        } else if (pos + 1u < m) {                        /* transpose with the next */
            uint32_t x = out[pos]; out[pos] = out[pos + 1u]; out[pos + 1u] = x;
        }
    }
    return m;
}

/* k swaps of two adjacent space-separated words. */
static size_t v_swaps(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                      const void *argp)
{
    static size_t ws[1024], wl[1024];
    static uint32_t tmp[MAXCP];
    size_t nw = 0, i, t, k = 0;
    unsigned swaps = *(const unsigned *)argp;
    for (i = 0; i < n && nw < 1024u; ) {
        while (i < n && in[i] == 0x20u) i++;
        if (i >= n) break;
        ws[nw] = i;
        while (i < n && in[i] != 0x20u) i++;
        wl[nw] = i - ws[nw];
        nw++;
    }
    for (t = 0; t < swaps && nw >= 2u; t++) {
        size_t a = (size_t)engram_rng_below(r, (uint64_t)(nw - 1u));
        size_t xs = ws[a], xl = wl[a];
        ws[a] = ws[a + 1u]; wl[a] = wl[a + 1u];
        ws[a + 1u] = xs; wl[a + 1u] = xl;
    }
    for (i = 0; i < nw; i++) {
        if (k && k < MAXCP) tmp[k++] = 0x20u;
        if (k + wl[i] > MAXCP) break;
        memcpy(tmp + k, in + ws[i], wl[i] * sizeof *in);
        k += wl[i];
    }
    if (k > cap) k = cap;
    memcpy(out, tmp, k * sizeof *out);
    return k;
}

/* A contiguous run of about `pct` percent of the codepoints, starting at a word boundary -- the part
 * of a paragraph a person half-remembers. */
static size_t v_fragment(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                         const void *argp)
{
    unsigned pct = *(const unsigned *)argp;
    size_t want = n * pct / 100u, start, tries;
    if (want < 8u) want = n < 8u ? n : 8u;
    start = 0;
    for (tries = 0; tries < 32u && n > want; tries++) {
        start = (size_t)engram_rng_below(r, (uint64_t)(n - want + 1u));
        if (start == 0u || in[start - 1u] == 0x20u) break;
    }
    if (want > cap) want = cap;
    memcpy(out, in + start, want * sizeof *out);
    return want;
}

typedef struct { unsigned pct, k; const alphabet *ab; } ft_arg;

/* A fragment, then typos inside it. */
static size_t v_fragtypo(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                         const void *argp)
{
    static uint32_t mid[MAXCP];
    const ft_arg *a = (const ft_arg *)argp;
    typo_arg t;
    size_t m = v_fragment(r, in, n, mid, MAXCP, &a->pct);
    if (!a->k) {
        if (m > cap) m = cap;
        memcpy(out, mid, m * sizeof *out);
        return m;
    }
    t.k = a->k;
    t.ab = a->ab;
    return v_typos(r, mid, m, out, cap, &t);
}

/* The paragraph's own COMPAT-normalised form: lowercase, unaccented, spaces collapsed. */
static size_t v_normalised(engram_rng *r, const uint32_t *in, size_t n, uint32_t *out, size_t cap,
                           const void *argp)
{
    static char raw[MAXB];
    char *norm = NULL;
    size_t nl = 0, rl, m;
    (void)r; (void)argp;
    rl = encode_all(in, n, raw, sizeof raw);
    if (engram_text_normalize(raw, rl, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT, &norm, &nl) != ENGRAM_OK)
        return 0;
    m = decode_all(norm, nl, out, cap);
    engram_free(norm);
    return m;
}

/* ==================================================================================================
 * QUERY SETS. Generated ONCE per corpus and shared by every configuration measured on it, so two
 * arms differ only in the encoder -- never in the queries they were asked.
 * ============================================================================================== */
typedef struct {
    size_t    n;          /* query texts                                            */
    size_t    nq;         /* source paragraphs; the texts of one source are adjacent */
    unsigned  per;        /* texts per source                                        */
    size_t   *src;        /* per text: the paragraph it was cut from                 */
    size_t   *off, *len;  /* per text: its bytes in buf                              */
    char     *buf;
    size_t    used, cap_buf, cap_src, cap_off, cap_len;
} qset;

static void qset_free(qset *q)
{
    engram_free(q->src);
    engram_free(q->off);
    engram_free(q->len);
    engram_free(q->buf);
    memset(q, 0, sizeof *q);
}

static int qset_add(qset *q, size_t src, const char *t, size_t tl)
{
    if (engram_grow((void **)&q->buf, &q->cap_buf, q->used + tl + 1u, 1u) != ENGRAM_OK ||
        engram_grow((void **)&q->src, &q->cap_src, q->n + 1u, sizeof *q->src) != ENGRAM_OK ||
        engram_grow((void **)&q->off, &q->cap_off, q->n + 1u, sizeof *q->off) != ENGRAM_OK ||
        engram_grow((void **)&q->len, &q->cap_len, q->n + 1u, sizeof *q->len) != ENGRAM_OK) return 0;
    memcpy(q->buf + q->used, t, tl);
    q->buf[q->used + tl] = 0;
    q->src[q->n] = src;
    q->off[q->n] = q->used;
    q->len[q->n] = tl;
    q->used += tl + 1u;
    q->n++;
    return 1;
}

/* One text per (source, generator) for sources first, first + stride, ...; generator g of source i is
 * seeded by (seed, g, i) alone, so the same paragraph always yields the same queries. */
static int qset_build(qset *q, const corpus *c, size_t first, size_t stride, unsigned ngen,
                      variant_fn fn, const void *const *args, uint64_t seed)
{
    static uint32_t in[MAXCP], out[MAXCP];
    static char txt[MAXB];
    size_t i;
    unsigned g;
    memset(q, 0, sizeof *q);
    q->per = ngen;
    for (i = first; i < c->n; i += stride) {
        size_t n = decode_all(c->line[i], c->len[i], in, MAXCP);
        for (g = 0; g < ngen; g++) {
            engram_rng r;
            size_t m, tl;
            engram_rng_seed(&r, engram_mix2(engram_mix2(seed, (uint64_t)g), (uint64_t)i));
            m = fn(&r, in, n, out, MAXCP, args[g]);
            tl = encode_all(out, m, txt, sizeof txt);
            if (!qset_add(q, i, txt, tl)) return 0;
        }
        q->nq++;
    }
    return 1;
}

/* The twelve fragment arms. */
static int qset_fragments(qset *q, const corpus *c, const alphabet *ab, size_t first, size_t stride)
{
    static ft_arg fa[NARM];
    const void *args[NARM];
    unsigned a;
    for (a = 0; a < NARM; a++) {
        fa[a].pct = ARM_PCT[a % 4u];
        fa[a].k = a / 4u;
        fa[a].ab = ab;
        args[a] = &fa[a];
    }
    return qset_build(q, c, first, stride, NARM, v_fragtypo, args, 0x46524147u);
}

/* ==================================================================================================
 * MEASUREMENT
 * ============================================================================================== */
typedef struct {
    size_t    n;
    uint8_t  *hit;            /* per query text: 1 = the source ranked first */
    double    mean;
    double    arm[NARM];
    unsigned  per;
    double    src_cos, best_other;
    size_t    refused;        /* query texts the encoder refused (E_SHORT): counted as misses */
    size_t    reached;        /* cascades: queries whose source reached the exact stage        */
} result;

static void result_free(result *r) { engram_free(r->hit); memset(r, 0, sizeof *r); }

/* Every query starts as a MISS: a path that skips a query (refused, or never reached stage 2) must
 * leave a 0 behind, not whatever the allocator returned -- engram_array does not clear, and the first
 * run of this suite reported a cascade "recall" of 1.42 because of exactly that. */
static int result_init(result *r, const qset *q)
{
    memset(r, 0, sizeof *r);
    r->n = q->n;
    r->per = q->per;
    r->hit = (uint8_t *)engram_array(q->n ? q->n : 1u, 1u);
    if (r->hit) memset(r->hit, 0, q->n ? q->n : 1u);
    return r->hit != NULL;
}

/* INSTRUMENT INTEGRITY: every hit is 0 or 1, so every recall lies in [0, 1]. Checked on every
 * result, because a measurement that can report 1.42 can also report a plausible 0.93 that is
 * equally wrong. */
static void result_finish(result *r)
{
    size_t t, bad = 0;
    unsigned a;
    double cnt[NARM];
    for (t = 0; t < r->n; t++) if (r->hit[t] > 1u) bad++;
    ET_CHECKF(bad == 0u, "INSTRUMENT: %lu hit values outside {0, 1}", (unsigned long)bad);
    memset(cnt, 0, sizeof cnt);
    memset(r->arm, 0, sizeof r->arm);
    r->mean = 0.0;
    for (t = 0; t < r->n; t++) {
        a = (unsigned)(t % r->per);
        if (a < NARM) { r->arm[a] += r->hit[t]; cnt[a] += 1.0; }
        r->mean += r->hit[t];
    }
    for (a = 0; a < NARM; a++) if (cnt[a] > 0.0) r->arm[a] /= cnt[a];
    if (r->n) { r->mean /= (double)r->n; r->src_cos /= (double)r->n; r->best_other /= (double)r->n; }
}

/* Eight lanes in a fixed order: vectorisable without -ffast-math, and still the same bits every run.
 * d is always a multiple of 8 (the encoder refuses anything else). */
static float fdot(const float *a, const float *b, unsigned d)
{
    float l[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    unsigned i, k;
    for (i = 0; i < d; i += 8u)
        for (k = 0; k < 8u; k++) l[k] += a[i + k] * b[i + k];
    return ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]));
}

static float *index_build(const engram_enc_cfg *cfg, unsigned d, const corpus *c)
{
    float *mat = (float *)engram_array(c->n * (size_t)d, sizeof *mat);
    size_t i;
    if (!mat) return NULL;
    for (i = 0; i < c->n; i++)
        if (engram_encode_d(cfg, c->line[i], c->len[i], mat + i * d, d, NULL) != ENGRAM_OK) {
            engram_free(mat);
            return NULL;
        }
    return mat;
}

/* Every paragraph's signature, ENGRAM_SIG_WORDS words each. */
static uint64_t *sig_build(const engram_enc_cfg *cfg, const corpus *c)
{
    uint64_t *sig = (uint64_t *)engram_array(c->n * (size_t)ENGRAM_SIG_WORDS, sizeof *sig);
    size_t i;
    if (!sig) return NULL;
    for (i = 0; i < c->n; i++)
        if (engram_encode_sig(cfg, c->line[i], c->len[i], sig + i * ENGRAM_SIG_WORDS, NULL) != ENGRAM_OK) {
            engram_free(sig);
            return NULL;
        }
    return sig;
}

/* Dense retrieval over the whole index. With stats, the mean source cosine and the mean best WRONG
 * cosine are gathered too (that needs the full scan; without stats a miss exits early). */
static int measure_dense(const engram_enc_cfg *cfg, unsigned d, const corpus *c, const float *mat,
                         const qset *q, result *r, int stats)
{
    static float v[ENGRAM_ENC_DMAX];
    size_t t, j;
    if (!result_init(r, q)) return 0;
    for (t = 0; t < q->n; t++) {
        size_t i = q->src[t];
        float s_src, best = -2.0f;
        int beaten = 0;
        if (engram_encode_d(cfg, q->buf + q->off[t], q->len[t], v, d, NULL) != ENGRAM_OK) {
            r->refused++;
            continue;
        }
        s_src = fdot(v, mat + i * d, d);
        for (j = 0; j < c->n; j++) {
            float s;
            if (j == i) continue;
            s = fdot(v, mat + j * d, d);
            if (s > s_src) { beaten = 1; if (!stats) break; }
            if (s > best) best = s;
        }
        r->hit[t] = (uint8_t)!beaten;
        r->src_cos += (double)s_src;
        r->best_other += (double)best;
    }
    result_finish(r);
    return 1;
}

static int double_desc(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x > y ? -1 : x < y ? 1 : 0;
}

/* THE CASCADE, as the store will run it. Stage 1 scores every paragraph -- by SIGNATURE containment
 * when `sigs` is given (the shipped design), else by dense cosine (the design T8 rejected, kept as a
 * control) -- and every paragraph scoring at least the C-th best goes on to stage 2, where the exact
 * score decides. The exact stage exists only for square-root shaping, so it runs with cfg's weights
 * and tf = SQRT whatever the dense stage uses. One pass also yields the dense stage's own recall@1
 * on the same queries (`dense`, may be NULL; needs `mat`). */
static int measure_cascade(const engram_enc_cfg *cfg, const corpus *c, const float *mat,
                           const uint64_t *sigs, const qset *q, unsigned C, result *r, result *dense)
{
    static float v[ENGRAM_D];
    double *sc = NULL, *sorted = NULL;
    engram_encq *eq = NULL;
    engram_enc_cfg ce = *cfg;
    size_t t, j;
    int ok = 0;
    ce.tf = ENGRAM_TF_SQRT;
    if (!result_init(r, q)) return 0;
    if (dense && !result_init(dense, q)) { result_free(r); return 0; }
    sc = (double *)engram_array(c->n ? c->n : 1u, sizeof *sc);
    sorted = (double *)engram_array(c->n ? c->n : 1u, sizeof *sorted);
    eq = (engram_encq *)engram_malloc(sizeof *eq);
    if (!sc || !sorted || !eq || !c->n || (!sigs && !mat) || (dense && !mat)) goto out;
    for (t = 0; t < q->n; t++) {
        size_t i = q->src[t];
        double thr, e_src, e;
        int beaten = 0;
        const char *qt = q->buf + q->off[t];
        if (mat) {
            if (engram_encode(cfg, qt, q->len[t], v, NULL) != ENGRAM_OK) {
                r->refused++;
                if (dense) dense->refused++;
                continue;
            }
            for (j = 0; j < c->n; j++) sc[j] = (double)fdot(v, mat + j * ENGRAM_D, ENGRAM_D);
            if (dense) {
                for (j = 0; j < c->n && !beaten; j++) if (j != i && sc[j] > sc[i]) beaten = 1;
                dense->hit[t] = (uint8_t)!beaten;
                beaten = 0;
            }
        }
        if (engram_encq_build(eq, &ce, qt, q->len[t]) != ENGRAM_OK) { r->refused++; continue; }
        if (sigs)
            for (j = 0; j < c->n; j++)
                if (engram_encq_sig_score(eq, sigs + j * ENGRAM_SIG_WORDS, &sc[j]) != ENGRAM_OK) goto out;
        memcpy(sorted, sc, c->n * sizeof *sc);
        qsort(sorted, c->n, sizeof *sorted, double_desc);
        thr = sorted[(C < c->n ? C : c->n) - 1u];
        if (sc[i] < thr) continue;                             /* the source never reached stage 2 */
        r->reached++;
        if (engram_encq_score(eq, c->line[i], c->len[i], &e_src) != ENGRAM_OK) goto out;
        for (j = 0; j < c->n && !beaten; j++) {
            if (j == i || sc[j] < thr) continue;
            if (engram_encq_score(eq, c->line[j], c->len[j], &e) != ENGRAM_OK) goto out;
            if (e > e_src) beaten = 1;
        }
        r->hit[t] = (uint8_t)!beaten;
    }
    result_finish(r);
    if (dense) result_finish(dense);
    ok = 1;
out:
    engram_free(sc);
    engram_free(sorted);
    engram_free(eq);
    if (!ok) { result_free(r); if (dense) result_free(dense); }
    return ok;
}

/* exact(query t, paragraph j). `tab` holds paragraph j's table when `tab_ok`; otherwise the query's
 * table is built in `scratch` and the paragraph streamed through it -- the same number by the
 * kernel's symmetry, which T14 proves. */
static int exact_pair(engram_encq *tab, int tab_ok, engram_encq *scratch, const engram_enc_cfg *cfg,
                      const char *qt, size_t ql, const char *dt, size_t dl, double *e)
{
    engram_rc rc;
    if (tab_ok) rc = engram_encq_score(tab, qt, ql, e);
    else {
        rc = engram_encq_build(scratch, cfg, qt, ql);
        if (rc == ENGRAM_OK) rc = engram_encq_score(scratch, dt, dl, e);
    }
    if (rc == ENGRAM_E_SHORT) { *e = 0.0; return 1; }
    return rc == ENGRAM_OK;
}

/* EXHAUSTIVE exact search -- the reference the cascade is judged against. Each PARAGRAPH's table is
 * built once and the short queries stream through it, instead of every paragraph streaming through
 * every query: the same scores (symmetry), an order of magnitude sooner. Pass 1 learns every query's
 * source score; pass 2 asks whether any other paragraph beats it. */
static int measure_exact_exhaustive(const engram_enc_cfg *cfg, const corpus *c, const qset *q,
                                    result *r, size_t *reversed)
{
    engram_encq *tab = (engram_encq *)engram_malloc(sizeof *tab);
    engram_encq *scratch = (engram_encq *)engram_malloc(sizeof *scratch);
    double *src = (double *)engram_array(q->n ? q->n : 1u, sizeof *src);
    uint8_t *beaten = (uint8_t *)engram_array(q->n ? q->n : 1u, 1u);
    size_t t, j;
    unsigned pass;
    int ok = 0;
    *reversed = 0;
    if (!tab || !scratch || !src || !beaten || !result_init(r, q)) goto out;
    memset(beaten, 0, q->n);
    for (pass = 0; pass < 2u; pass++) {
        for (j = 0; j < c->n; j++) {
            engram_rc rc = engram_encq_build(tab, cfg, c->line[j], c->len[j]);
            int tab_ok = (rc == ENGRAM_OK);
            if (rc == ENGRAM_E_FULL) { if (pass == 0u) (*reversed)++; }
            else if (rc != ENGRAM_OK) goto out;
            for (t = 0; t < q->n; t++) {
                double e;
                if (pass == 0u ? (q->src[t] != j) : (q->src[t] == j || beaten[t])) continue;
                if (!exact_pair(tab, tab_ok, scratch, cfg, q->buf + q->off[t], q->len[t],
                                c->line[j], c->len[j], &e)) goto out;
                if (pass == 0u) src[t] = e;
                else if (e > src[t]) beaten[t] = 1;
            }
        }
    }
    for (t = 0; t < q->n; t++) r->hit[t] = (uint8_t)!beaten[t];
    result_finish(r);
    ok = 1;
out:
    engram_free(tab);
    engram_free(scratch);
    engram_free(src);
    engram_free(beaten);
    return ok;
}

/* ---- THE PAIRED MARGIN ------------------------------------------------------------------------ */
typedef struct { double diff, margin; } paired;

static paired compare(const result *base, const result *alt)
{
    paired p;
    size_t nq = base->per ? base->n / base->per : 0u, g, t;
    double s = 0.0, sq = 0.0, mean, var;
    p.diff = alt->mean - base->mean;
    p.margin = 1.0;
    if (!nq || base->n != alt->n || base->per != alt->per) return p;
    for (g = 0; g < nq; g++) {
        double dsum = 0.0;
        for (t = g * base->per; t < (g + 1u) * base->per; t++)
            dsum += (double)alt->hit[t] - (double)base->hit[t];
        dsum /= (double)base->per;
        s += dsum;
        sq += dsum * dsum;
    }
    mean = s / (double)nq;
    var = nq > 1u ? (sq - (double)nq * mean * mean) / (double)(nq - 1u) : 0.0;
    if (var < 0.0) var = 0.0;
    p.margin = 2.0 * sqrt(var / (double)nq) + 1.0 / (double)base->n;
    return p;
}

/* The standard error of a result's mean, clustered by source paragraph (see THE PAIRED MARGIN). */
static double clustered_se(const result *r)
{
    size_t nq = r->per ? r->n / r->per : 0u, g, t;
    double s = 0.0, sq = 0.0, mean, var;
    if (nq < 2u) return 1.0;
    for (g = 0; g < nq; g++) {
        double m = 0.0;
        for (t = g * r->per; t < (g + 1u) * r->per; t++) m += (double)r->hit[t];
        m /= (double)r->per;
        s += m;
        sq += m * m;
    }
    mean = s / (double)nq;
    var = (sq - (double)nq * mean * mean) / (double)(nq - 1u);
    return var > 0.0 ? sqrt(var / (double)nq) : 0.0;
}

static void print_result(const char *name, const result *r)
{
    unsigned a;
    printf("       %-34s %6.4f |", name, r->mean);
    if (r->per == NARM)
        for (a = 0; a < NARM; a++) printf("%s%5.3f", (a % 4u) ? " " : "  ", r->arm[a]);
    else
        printf(" cos(src) %6.4f  best-other %6.4f", r->src_cos, r->best_other);
    printf("  n=%lu", (unsigned long)r->n);
    if (r->refused) printf("  (%lu refused)", (unsigned long)r->refused);
    printf("\n");
    fflush(stdout);
}

/* alt must NOT beat base beyond the paired margin. A "no gain" verdict from an instrument that
 * cannot show a gain is worthless, so the gate refuses to certify on a saturated base (R6). */
static void gate_no_gain(const char *what, const char *name, const result *base, const result *alt)
{
    paired p = compare(base, alt);
    printf("         %-40s %+.4f (margin %.4f)  must not gain\n", name, p.diff, p.margin);
    ET_CHECKF(base->mean < SATURATED, "%s: the instrument is saturated (%.4f) -- a 'no gain' "
              "verdict from it would be vacuous", what, base->mean);
    ET_CHECKF(p.diff <= p.margin, "%s: '%s' BEATS the default by %+.4f > margin %.4f",
              what, name, p.diff, p.margin);
}

/* Reported, deliberately NOT gated -- and the line says so, so it cannot be mistaken for a pass. */
static void report_diff(const char *name, const result *base, const result *alt)
{
    paired p = compare(base, alt);
    printf("         %-40s %+.4f (margin %.4f)  reported, not gated\n", name, p.diff, p.margin);
}

/* alt must LOSE to base beyond the paired margin -- a gate shown able to see a difference. */
static void gate_loses(const char *what, const char *name, const result *base, const result *alt)
{
    paired p = compare(base, alt);
    printf("         %-40s %+.4f (margin %.4f)  must lose\n", name, p.diff, p.margin);
    ET_CHECKF(p.diff < -p.margin, "%s: '%s' does not lose beyond the margin (%+.4f, margin %.4f)",
              what, name, p.diff, p.margin);
}

/* A configuration measured on a query set: encode the index, run, print, free the index. */
static int run_config(const char *name, const engram_enc_cfg *cfg, unsigned d, const corpus *c,
                      const qset *q, result *r)
{
    float *mat = index_build(cfg, d, c);
    int ok;
    memset(r, 0, sizeof *r);
    if (!mat) { ET_CHECKF(0, "index for '%s' did not build", name); return 0; }
    ok = measure_dense(cfg, d, c, mat, q, r, 0);
    engram_free(mat);
    if (!ok) { ET_CHECKF(0, "measurement of '%s' failed", name); return 0; }
    print_result(name, r);
    return 1;
}

/* The shipped retrieval: signature first stage, exact second. `dense` receives the dense stage's own
 * recall@1 on the same queries. With `dense_first`, the first stage is dense cosine instead -- the
 * control T8 compares against. */
static int run_cascade_config(const char *name, const engram_enc_cfg *cfg, const corpus *c,
                              const qset *q, int dense_first, result *casc, result *dense)
{
    float *mat = index_build(cfg, ENGRAM_D, c);
    uint64_t *sigs = dense_first ? NULL : sig_build(cfg, c);
    char label[96];
    int ok = 0;
    memset(casc, 0, sizeof *casc);
    if (dense) memset(dense, 0, sizeof *dense);
    if (!mat || (!dense_first && !sigs)) { ET_CHECKF(0, "index for '%s' did not build", name); goto out; }
    ok = measure_cascade(cfg, c, mat, sigs, q, CASCADE_C, casc, dense);
    if (!ok) { ET_CHECKF(0, "cascade measurement of '%s' failed", name); goto out; }
    snprintf(label, sizeof label, "%s  [cascade%s]", name, dense_first ? ", dense first" : "");
    print_result(label, casc);
    printf("         source reached the exact stage for %.4f of queries (top %u of %lu)\n",
           casc->n ? (double)casc->reached / (double)casc->n : 0.0, CASCADE_C, (unsigned long)c->n);
    if (dense) {
        snprintf(label, sizeof label, "%s  [dense]", name);
        print_result(label, dense);
    }
out:
    engram_free(mat);
    engram_free(sigs);
    return ok;
}

static void cfg_trace(engram_enc_cfg *cfg)
{
    /* VECTRA TRACE's weighting in this encoder: characters 1, every word 4, every pair 6, linear. */
    engram_enc_cfg_default(cfg);
    cfg->w_char3 = 1.0f;
    cfg->w_word1 = cfg->w_ideo1 = 4.0f;
    cfg->w_word2 = cfg->w_ideo2 = 6.0f;
    cfg->tf = ENGRAM_TF_LINEAR;
}

/* ==================================================================================================
 * THE SUITE -- CONTRACT
 * ============================================================================================== */
static void test_contract(void)
{
    float v[ENGRAM_D], w[ENGRAM_D];
    engram_enc_cfg c;
    engram_enc_stats st;
    unsigned i;
    int zero;
    char *big;
    size_t bl, k;
    uint64_t calls;

    ET_SECTION("contract: empty and blank text produce no vector (E_SHORT), and zeros");
    v[0] = 5.0f;
    ET_RC(engram_encode(NULL, "", 0u, v, NULL), ENGRAM_E_SHORT);
    for (zero = 1, i = 0; i < ENGRAM_D; i++) if (v[i] != 0.0f) zero = 0;
    ET_CHECK(zero);
    ET_RC(engram_encode(NULL, " \t\n ", 4u, v, NULL), ENGRAM_E_SHORT);
    ET_RC(engram_encode(NULL, "''\xE2\x80\x8D", 5u, v, NULL), ENGRAM_E_SHORT);   /* only IGNOREs */

    ET_SECTION("contract: one letter is enough to encode");
    ET_OK(engram_encode(NULL, "a", 1u, v, &st));
    ET_CHECK(fabs(engram_vec_norm(v, ENGRAM_D) - 1.0) < 1e-5);
    ET_EQ_U64(st.words, 1u);
    ET_EQ_U64(st.f_char3, 1u);                         /* " a " */
    ET_EQ_U64(st.f_char4, 0u);
    ET_EQ_U64(st.f_word1, 0u);                         /* alphabetic words weigh 0 by default */

    ET_SECTION("contract: ideographs are words of one character, with their own weights");
    ET_OK(engram_encode(NULL, "\xE4\xB8\xAD\xE6\x96\x87", 6u, v, &st));          /* two Han */
    ET_EQ_U64(st.words, 2u);
    ET_EQ_U64(st.ideographs, 2u);
    ET_EQ_U64(st.f_ideo1, 2u);
    ET_EQ_U64(st.f_ideo2, 1u);
    ET_EQ_U64(st.f_word1, 0u);
    ET_EQ_U64(st.f_word2, 0u);
    ET_OK(engram_encode(NULL, "abc\xE4\xB8\xAD", 6u, v, &st));                   /* word, then Han */
    ET_EQ_U64(st.words, 2u);
    ET_EQ_U64(st.ideographs, 1u);
    ET_EQ_U64(st.f_ideo1, 1u);
    ET_EQ_U64(st.f_ideo2, 0u);          /* the seam pair is not two ideographs: it takes w_word2 = 0 */
    ET_EQ_U64(st.f_word2, 0u);
    engram_enc_cfg_default(&c);
    c.w_word2 = 1.0f;
    ET_OK(engram_encode(&c, "abc\xE4\xB8\xAD", 6u, v, &st));
    ET_EQ_U64(st.f_word2, 1u);                     /* ...and it is the SEAM pair that takes it */
    ET_EQ_U64(st.f_ideo2, 0u);
    engram_enc_cfg_default(&c);
    c.w_char3 = c.w_char4 = 0.0f;                  /* ideographic features only: still a valid cfg */
    ET_OK(engram_enc_cfg_check(&c));
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_SHORT);
    ET_OK(engram_encode(&c, "\xE4\xB8\xAD", 3u, v, NULL));

    ET_SECTION("contract: case and accents cannot be told apart under COMPAT");
    ET_OK(engram_encode(NULL, "Caf\xC3\xA9 CR\xC3\x88ME", 12u, v, NULL));
    ET_OK(engram_encode(NULL, "cafe creme", 10u, w, NULL));
    ET_CHECK(memcmp(v, w, sizeof v) == 0);

    ET_SECTION("contract: with alphabetic words at weight 0, pair ORDER cannot move a Latin vector");
    engram_enc_cfg_default(&c);
    c.ordered_pairs = 1;
    ET_OK(engram_encode(NULL, "the quick brown fox", 19u, v, NULL));
    ET_OK(engram_encode(&c, "the quick brown fox", 19u, w, NULL));
    ET_CHECK(memcmp(v, w, sizeof v) == 0);

    ET_SECTION("contract: invalid UTF-8 -- STRICT refuses with zeros, REPLACE encodes and counts");
    engram_enc_cfg_default(&c);
    c.utf8 = ENGRAM_UTF8_STRICT;
    ET_RC(engram_encode(&c, "abc\xC0\xAF", 5u, v, &st), ENGRAM_E_UTF8);
    for (zero = 1, i = 0; i < ENGRAM_D; i++) if (v[i] != 0.0f) zero = 0;
    ET_CHECK(zero);
    c.utf8 = ENGRAM_UTF8_REPLACE;
    ET_OK(engram_encode(&c, "abc\xC0\xAF", 5u, v, &st));
    ET_EQ_U64(st.invalid_utf8, 2u);

    ET_SECTION("contract: bad arguments and bad configurations are refused");
    ET_RC(engram_encode_d(NULL, "abc", 3u, v, 60u, NULL), ENGRAM_E_ARG);      /* below 64     */
    ET_RC(engram_encode_d(NULL, "abc", 3u, v, 100u, NULL), ENGRAM_E_ARG);     /* not /8       */
    ET_RC(engram_encode_d(NULL, "abc", 3u, v, 8192u, NULL), ENGRAM_E_ARG);    /* above DMAX   */
    ET_RC(engram_encode(NULL, "abc", 3u, NULL, NULL), ENGRAM_E_ARG);
    ET_RC(engram_encode(NULL, NULL, 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.w_char3 = -1.0f;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.w_word1 = (float)NAN;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.w_ideo2 = (float)INFINITY;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.w_ideo1 = -0.5f;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c);
    c.w_char3 = c.w_char4 = c.w_word1 = c.w_word2 = c.w_ideo1 = c.w_ideo2 = 0.0f;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.fold = (engram_fold)7;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.tf = (engram_tf)9;
    ET_RC(engram_encode(&c, "abc", 3u, v, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c); c.w_char3 = -0.0f;                    /* -0 IS non-negative */
    ET_OK(engram_enc_cfg_check(&c));

    ET_SECTION("T2: a 4 MB document encodes with ZERO allocations");
    bl = (size_t)4u << 20;
    big = (char *)engram_malloc(bl + 1u);
    ET_CHECK(big != NULL);
    if (big) {
        const char *s = "The quick brown fox jumps over the lazy dog, and then again. ";
        size_t sl = strlen(s);
        for (k = 0; k + sl <= bl; k += sl) memcpy(big + k, s, sl);
        calls = engram_alloc_calls();
        ET_OK(engram_encode(NULL, big, k, v, &st));
        ET_EQ_U64(engram_alloc_calls(), calls);
        ET_CHECK(fabs(engram_vec_norm(v, ENGRAM_D) - 1.0) < 1e-5);
        printf("       4 MB: %llu words, %llu char3, 0 allocations\n",
               (unsigned long long)st.words, (unsigned long long)st.f_char3);
        engram_free(big);
    }

    ET_SECTION("geometry fingerprint: stable, and moved by every parameter that moves the vectors");
    {
        uint64_t g0 = engram_enc_geometry(NULL, ENGRAM_D);
        engram_enc_cfg_default(&c);
        ET_EQ_U64(engram_enc_geometry(&c, ENGRAM_D), g0);
        ET_CHECK(engram_enc_geometry(&c, 256u) != g0);
        c.seed ^= 1u;        ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.w_char3 = 2.0f;    ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.w_word2 = 5.0f;    ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.w_ideo1 = 8.0f;    ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.w_ideo2 = 8.0f;    ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.ordered_pairs = 1; ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.fold = ENGRAM_FOLD_CASE; ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.tf = ENGRAM_TF_LINEAR;   ET_CHECK(engram_enc_geometry(&c, ENGRAM_D) != g0); engram_enc_cfg_default(&c);
        c.utf8 = ENGRAM_UTF8_STRICT;                   /* policy does NOT move vectors of valid text */
        ET_EQ_U64(engram_enc_geometry(&c, ENGRAM_D), g0);
        printf("       default geometry 0x%016llX\n", (unsigned long long)g0);
    }
}

/* ==================================================================================================
 * T14 -- THE EXACT STAGE
 * ============================================================================================== */

/* THE INDEPENDENT REFERENCE. For lowercase ASCII words separated by single spaces the encoder's
 * stream is " w1 w2 ... wn " exactly, and with the default weights its features are every
 * 3-character substring of that string at weight 1/16 and every 4-character substring at weight 1
 * (alphabetic words weigh 0). This computes the Bhattacharyya coefficient of the two weighted
 * substring multisets by sorting -- no hashing, no table, no code shared with the encoder. */
static int u64_cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static size_t ref_grams(const char *s, uint64_t *out)
{
    size_t L = strlen(s), i, n = 0, g, j;
    for (g = 3u; g <= 4u; g++)
        for (i = 0; i + g <= L; i++) {
            uint64_t k = (uint64_t)g << 56;
            for (j = 0; j < g; j++) k |= (uint64_t)(unsigned char)s[i + j] << (8u * j);
            out[n++] = k;
        }
    qsort(out, n, sizeof *out, u64_cmp);
    return n;
}

static double ref_weight(uint64_t gram) { return (gram >> 56) == 3u ? 0.0625 : 1.0; }

static double ref_bc(const char *a, const char *b)
{
    static uint64_t ga[512], gb[512];
    size_t na = ref_grams(a, ga), nb = ref_grams(b, gb), i = 0, j = 0;
    double num = 0.0, ma = 0.0, mb = 0.0;
    for (i = 0; i < na; i++) ma += ref_weight(ga[i]);
    for (j = 0; j < nb; j++) mb += ref_weight(gb[j]);
    i = j = 0;
    while (i < na && j < nb) {
        if (ga[i] < gb[j]) i++;
        else if (gb[j] < ga[i]) j++;
        else {
            uint64_t k = ga[i];
            double ca = 0.0, cb = 0.0;
            while (i < na && ga[i] == k) { ca += ref_weight(k); i++; }
            while (j < nb && gb[j] == k) { cb += ref_weight(k); j++; }
            num += sqrt(ca * cb);
        }
    }
    return (ma > 0.0 && mb > 0.0) ? num / sqrt(ma * mb) : 0.0;
}

/* " w1 w2 ... " over a six-letter alphabet, so shared substrings are common. */
static void rand_words(engram_rng *r, char *out, size_t cap)
{
    unsigned nw = 1u + (unsigned)engram_rng_below(r, 8u), w, k;
    size_t n = 0;
    out[n++] = ' ';
    for (w = 0; w < nw && n + 10u < cap; w++) {
        unsigned wl = 1u + (unsigned)engram_rng_below(r, 6u);
        for (k = 0; k < wl; k++) out[n++] = (char)('a' + (int)engram_rng_below(r, 6u));
        out[n++] = ' ';
    }
    out[n] = 0;
}

static void test_exact(void)
{
    engram_encq *q = (engram_encq *)engram_malloc(sizeof *q);
    engram_encq *q2 = (engram_encq *)engram_malloc(sizeof *q2);
    engram_enc_cfg c;
    double s = 0.0, s2 = 0.0;
    uint64_t calls;
    char a[128], b[128];
    unsigned i, bad = 0, asym = 0;
    double worst = 0.0;
    engram_rng r;

    ET_SECTION("T14: exact stage -- refusals");
    ET_CHECK(q && q2);
    if (!q || !q2) { engram_free(q); engram_free(q2); return; }
    ET_RC(engram_encq_build(NULL, NULL, "abc", 3u), ENGRAM_E_ARG);
    ET_RC(engram_encq_build(q, NULL, NULL, 3u), ENGRAM_E_ARG);
    ET_RC(engram_encq_score(q, "abc", 3u, &s), ENGRAM_E_STATE);          /* failed build: not ready */
    ET_RC(engram_encq_build(q, NULL, "", 0u), ENGRAM_E_SHORT);
    ET_RC(engram_encq_score(q, "abc", 3u, &s), ENGRAM_E_STATE);
    engram_enc_cfg_default(&c);
    c.tf = ENGRAM_TF_LINEAR;                                             /* no closed-form norm */
    ET_RC(engram_encq_build(q, &c, "abc", 3u), ENGRAM_E_ARG);
    c.tf = ENGRAM_TF_QUARTER;
    ET_RC(engram_encq_build(q, &c, "abc", 3u), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c);
    c.w_word1 = (float)NAN;
    ET_RC(engram_encq_build(q, &c, "abc", 3u), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c);
    c.utf8 = ENGRAM_UTF8_STRICT;
    ET_RC(engram_encq_build(q, &c, "ab\xC0", 3u), ENGRAM_E_UTF8);
    ET_OK(engram_encq_build(q, &c, "abc", 3u));
    s = 7.0;
    ET_RC(engram_encq_score(q, "ab\xC0", 3u, &s), ENGRAM_E_UTF8);        /* the query's policy rules */
    ET_CHECK(s == 0.0);
    ET_RC(engram_encq_score(q, "abc", 3u, NULL), ENGRAM_E_ARG);
    ET_RC(engram_encq_score(NULL, "abc", 3u, &s), ENGRAM_E_ARG);
    s = 7.0;
    ET_RC(engram_encq_score(q, "   ", 3u, &s), ENGRAM_E_SHORT);
    ET_CHECK(s == 0.0);

    ET_SECTION("T14: a query too long for the table is refused, and stays refused");
    {
        static char longq[6000];
        size_t n = 0;
        engram_rng_seed(&r, 0x10A6u);
        while (n + 2u < sizeof longq - 1u) {
            longq[n++] = (char)('a' + (int)engram_rng_below(&r, 26u));
            if (engram_rng_below(&r, 7u) == 0u) longq[n++] = ' ';
        }
        longq[n] = 0;
        ET_RC(engram_encq_build(q, NULL, longq, n), ENGRAM_E_FULL);
        ET_RC(engram_encq_score(q, "abc", 3u, &s), ENGRAM_E_STATE);
    }

    ET_SECTION("T14: identity, folding, and nothing shared scores exactly 0");
    ET_OK(engram_encq_build(q, NULL, "the quick brown fox", 19u));
    ET_OK(engram_encq_score(q, "the quick brown fox", 19u, &s));
    ET_CHECKF(fabs(s - 1.0) < 1e-12, "identity scored %.17g", s);
    ET_OK(engram_encq_score(q, "The QUICK Brown fox", 19u, &s));
    ET_CHECKF(fabs(s - 1.0) < 1e-12, "folded identity scored %.17g", s);
    ET_OK(engram_encq_score(q, "zzz yyy", 7u, &s));
    ET_CHECK(s == 0.0);

    ET_SECTION("T14: equal to an independent reference on 2,000 random pairs (to 1e-12), symmetric");
    engram_rng_seed(&r, 0xB4A7u);
    for (i = 0; i < 2000u; i++) {
        double ref;
        rand_words(&r, a, sizeof a);
        if (engram_rng_below(&r, 3u) == 0u) memcpy(b, a, sizeof a);   /* identical sometimes */
        else rand_words(&r, b, sizeof b);
        ref = ref_bc(a, b);
        if (engram_encq_build(q, NULL, a, strlen(a)) != ENGRAM_OK ||
            engram_encq_score(q, b, strlen(b), &s) != ENGRAM_OK ||
            engram_encq_build(q2, NULL, b, strlen(b)) != ENGRAM_OK ||
            engram_encq_score(q2, a, strlen(a), &s2) != ENGRAM_OK) { bad++; continue; }
        if (fabs(s - ref) > worst) worst = fabs(s - ref);
        if (fabs(s - ref) > 1e-12) bad++;
        if (fabs(s - s2) > 1e-12) asym++;
    }
    printf("       worst |exact - reference| = %.3g over 2000 pairs\n", worst);
    ET_EQ_U64(bad, 0u);
    ET_EQ_U64(asym, 0u);

    ET_SECTION("T14: an ideographic case by hand -- exactly 128/393");
    /* "中文" and "中国": features " 中文", "中文 " (3-grams, 1/16 each), " 中文 " (4-gram, 1), 中 and
     * 文 (16 each), the pair (16): mass 1/8 + 1 + 48 = 393/8 each. Shared: only the unigram 中,
     * sqrt(16 * 16) = 16. So 16 / (393/8) = 128/393. */
    ET_OK(engram_encq_build(q, NULL, "\xE4\xB8\xAD\xE6\x96\x87", 6u));
    ET_EQ_U64(q->n, 6u);
    ET_OK(engram_encq_score(q, "\xE4\xB8\xAD\xE5\x9B\xBD", 6u, &s));
    ET_CHECKF(fabs(s - 128.0 / 393.0) < 1e-15, "scored %.17g, expected 128/393", s);

    ET_SECTION("T14: deterministic to the bit, and a 4 MB document scores with ZERO allocations");
    {
        size_t bl = (size_t)4u << 20, k;
        char *big = (char *)engram_malloc(bl + 1u);
        ET_CHECK(big != NULL);
        if (big) {
            const char *t = "The quick brown fox jumps over the lazy dog, and then again. ";
            size_t tl = strlen(t);
            for (k = 0; k + tl <= bl; k += tl) memcpy(big + k, t, tl);
            calls = engram_alloc_calls();
            ET_OK(engram_encq_build(q, NULL, "lazy dog jumps", 14u));
            ET_OK(engram_encq_score(q, big, k, &s));
            ET_OK(engram_encq_score(q, big, k, &s2));
            ET_EQ_U64(engram_alloc_calls(), calls);
            ET_CHECK(memcmp(&s, &s2, sizeof s) == 0);
            ET_CHECK(s > 0.0 && s < 1.0);
            printf("       'lazy dog jumps' against 4 MB of that sentence: %.6f, 0 allocations\n", s);
            engram_free(big);
        }
    }
    engram_free(q);
    engram_free(q2);
}

/* ==================================================================================================
 * T15 -- THE SIGNATURE
 * ============================================================================================== */
static unsigned popcount64(uint64_t x)
{
    unsigned n = 0;
    while (x) { x &= x - 1u; n++; }
    return n;
}

static unsigned sig_popcount(const uint64_t *sig)
{
    unsigned w, n = 0;
    for (w = 0; w < ENGRAM_SIG_WORDS; w++) n += popcount64(sig[w]);
    return n;
}

static void test_signature(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    static uint64_t sig[ENGRAM_SIG_WORDS], sig2[ENGRAM_SIG_WORDS];
    engram_encq *q = (engram_encq *)engram_malloc(sizeof *q);
    engram_enc_cfg c;
    engram_enc_stats st;
    double s = 0.0;
    unsigned li, i, bad = 0, total = 0, overfull = 0;
    uint64_t sp = 0x5167A7u, calls;
    engram_rng r;
    char a[128];

    ET_SECTION("T15: signature -- refusals leave no bits behind");
    ET_CHECK(q != NULL);
    if (!q) return;
    ET_RC(engram_encode_sig(NULL, "abc", 3u, NULL, NULL), ENGRAM_E_ARG);
    sig[0] = ~(uint64_t)0;
    ET_RC(engram_encode_sig(NULL, "", 0u, sig, NULL), ENGRAM_E_SHORT);
    ET_EQ_U64(sig_popcount(sig), 0u);
    ET_RC(engram_encode_sig(NULL, NULL, 3u, sig, NULL), ENGRAM_E_ARG);
    engram_enc_cfg_default(&c);
    c.utf8 = ENGRAM_UTF8_STRICT;
    ET_RC(engram_encode_sig(&c, "abc def \xC0", 9u, sig, NULL), ENGRAM_E_UTF8);
    ET_EQ_U64(sig_popcount(sig), 0u);                  /* the bits of "abc def " were set, then wiped */
    engram_enc_cfg_default(&c);
    c.w_char4 = (float)NAN;
    ET_RC(engram_encode_sig(&c, "abc", 3u, sig, NULL), ENGRAM_E_ARG);
    ET_RC(engram_encq_sig_score(q, sig, NULL), ENGRAM_E_ARG);
    ET_RC(engram_encq_sig_score(NULL, sig, &s), ENGRAM_E_ARG);
    memset(q, 0, sizeof *q);
    ET_RC(engram_encq_sig_score(q, sig, &s), ENGRAM_E_STATE);

    ET_SECTION("T15: at most K bits per feature; a single letter still has a signature");
    ET_OK(engram_encode_sig(NULL, "a", 1u, sig, &st));
    ET_CHECK(sig_popcount(sig) >= 1u && sig_popcount(sig) <= ENGRAM_SIG_K);
    ET_OK(engram_encode_sig(NULL, "the quick brown fox", 19u, sig, &st));
    ET_CHECK(sig_popcount(sig) <= ENGRAM_SIG_K * (unsigned)(st.f_char3 + st.f_char4 + st.f_word1 +
                                                            st.f_word2 + st.f_ideo1 + st.f_ideo2));

    ET_SECTION("T15: NO FALSE NEGATIVES -- every text scores exactly 1 against its own signature");
    engram_rng_seed(&r, 0x5165u);
    for (i = 0; i < 2000u; i++) {
        rand_words(&r, a, sizeof a);
        if (engram_encode_sig(NULL, a, strlen(a), sig, NULL) != ENGRAM_OK ||
            engram_encq_build(q, NULL, a, strlen(a)) != ENGRAM_OK ||
            engram_encq_sig_score(q, sig, &s) != ENGRAM_OK || s != 1.0) bad++;
        total++;
    }
    for (li = 0; li < 5u; li++) {
        corpus cp;
        size_t j;
        double fill = 0.0;
        if (!corpus_load(&cp, langs[li])) { ET_CHECK(0); corpus_free(&cp); continue; }
        for (j = 0; j < cp.n; j++) {
            engram_rc rc;
            if (engram_encode_sig(NULL, cp.line[j], cp.len[j], sig, NULL) != ENGRAM_OK) { bad++; continue; }
            fill += (double)sig_popcount(sig) / (double)ENGRAM_SIG_BITS;
            sp = engram_mix2(sp, engram_hash_bytes(sig, sizeof sig, (uint64_t)total));
            rc = engram_encq_build(q, NULL, cp.line[j], cp.len[j]);
            if (rc == ENGRAM_E_FULL) { overfull++; continue; }
            if (rc != ENGRAM_OK || engram_encq_sig_score(q, sig, &s) != ENGRAM_OK || s != 1.0) bad++;
            total++;
        }
        printf("       %s: mean signature fill %.3f over %lu paragraphs\n",
               langs[li], cp.n ? fill / (double)cp.n : 0.0, (unsigned long)cp.n);
        corpus_free(&cp);
    }
    printf("       %u texts scored against their own signatures, %u did not score exactly 1 "
           "(%u too long for a query table)\n", total, bad, overfull);
    ET_EQ_U64(bad, 0u);
    printf("SIGPRINT %016llX\n", (unsigned long long)sp);

    ET_SECTION("T15: deterministic to the bit, and a 4 MB document signs with ZERO allocations");
    {
        size_t bl = (size_t)4u << 20, k;
        char *big = (char *)engram_malloc(bl + 1u);
        ET_CHECK(big != NULL);
        if (big) {
            const char *t = "The quick brown fox jumps over the lazy dog, and then again. ";
            size_t tl = strlen(t);
            for (k = 0; k + tl <= bl; k += tl) memcpy(big + k, t, tl);
            calls = engram_alloc_calls();
            ET_OK(engram_encode_sig(NULL, big, k, sig, NULL));
            ET_OK(engram_encode_sig(NULL, big, k, sig2, NULL));
            ET_EQ_U64(engram_alloc_calls(), calls);
            ET_CHECK(memcmp(sig, sig2, sizeof sig) == 0);
            printf("       4 MB of one repeated sentence: %u bits set\n", sig_popcount(sig));
            engram_free(big);
        }
    }
    engram_free(q);
}

/* ==================================================================================================
 * T1/T2 AT SCALE, AND THE CROSS-PLATFORM PRINTS
 * ============================================================================================== */
static void test_fingerprint(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    float v[ENGRAM_D], w[ENGRAM_D];
    uint64_t fp = 0x0123456789ABCDEFull, xp = 0xFEDCBA9876543210ull;
    unsigned li, nvec = 0, badnorm = 0, nondet = 0, nfull = 0;
    size_t i, j;
    uint64_t calls;
    double t0, secs, bytes = 0.0, xbytes = 0.0, xsecs;
    engram_encq *q = (engram_encq *)engram_malloc(sizeof *q);

    ET_SECTION("T1/T2: every paragraph of every language -- unit norm, deterministic, no allocation");
    ET_CHECK(q != NULL);
    if (!q) return;
    t0 = (double)engram_now_ns();
    for (li = 0; li < sizeof langs / sizeof langs[0]; li++) {
        corpus c;
        ET_CHECKF(corpus_load(&c, langs[li]), "corpus_%s.txt did not load", langs[li]);
        if (!c.n) { corpus_free(&c); continue; }
        calls = engram_alloc_calls();
        for (i = 0; i < c.n; i++) {
            if (engram_encode(NULL, c.line[i], c.len[i], v, NULL) != ENGRAM_OK) { badnorm++; continue; }
            if (fabs(engram_vec_norm(v, ENGRAM_D) - 1.0) >= 1e-5) badnorm++;
            if (engram_encode(NULL, c.line[i], c.len[i], w, NULL) != ENGRAM_OK ||
                memcmp(v, w, sizeof v) != 0) nondet++;
            fp = engram_mix2(fp, engram_hash_bytes(v, sizeof v, (uint64_t)nvec));
            bytes += (double)c.len[i];
            nvec++;
        }
        ET_EQ_U64(engram_alloc_calls(), calls);
        corpus_free(&c);
    }
    secs = ((double)engram_now_ns() - t0) / 1e9;
    ET_EQ_U64(badnorm, 0u);
    ET_EQ_U64(nondet, 0u);
    ET_CHECK(nvec > 1500u);
    printf("       %u vectors across 5 languages; dense encoding %.1f MB/s\n",
           nvec, secs > 0.0 ? 2.0 * bytes / secs / 1e6 : 0.0);
    printf("FINGERPRINT %016llX\n", (unsigned long long)fp);

    ET_SECTION("T14/R5: exact scores among the first 40 paragraphs of each language, hashed");
    t0 = (double)engram_now_ns();
    for (li = 0; li < sizeof langs / sizeof langs[0]; li++) {
        corpus c;
        size_t m;
        if (!corpus_load(&c, langs[li])) { ET_CHECK(0); corpus_free(&c); continue; }
        m = c.n < 40u ? c.n : 40u;
        for (i = 0; i < m; i++) {
            engram_rc rc = engram_encq_build(q, NULL, c.line[i], c.len[i]);
            if (rc == ENGRAM_E_FULL) { nfull++; continue; }        /* a long paragraph: no table */
            if (rc != ENGRAM_OK) { ET_CHECK(0); continue; }
            for (j = 0; j < m; j++) {
                double s;
                if (engram_encq_score(q, c.line[j], c.len[j], &s) != ENGRAM_OK) { ET_CHECK(0); continue; }
                xp = engram_mix2(xp, engram_hash_bytes(&s, sizeof s, (uint64_t)(i * 64u + j)));
                xbytes += (double)c.len[j];
            }
        }
        corpus_free(&c);
    }
    xsecs = ((double)engram_now_ns() - t0) / 1e9;
    printf("       exact scoring %.1f MB/s of document streamed (%u paragraphs too long for a table)\n",
           xsecs > 0.0 ? xbytes / xsecs / 1e6 : 0.0, nfull);
    printf("EXACTPRINT %016llX\n", (unsigned long long)xp);
    engram_free(q);
}

/* ==================================================================================================
 * T3-T6, T10 -- WHOLE PARAGRAPHS (claims stated in advance; floors, not comparisons)
 * ============================================================================================== */
static void test_whole(void)
{
    static const char *names[] = { "1 typo", "3 typos", "5 typos", "10 typos (reported)" };
    static const double floors[] = { 0.98, 0.95, 0.90, 0.0 };
    corpus c;
    alphabet ab;
    engram_enc_cfg cfg;
    float *mat;
    size_t stride = g_quick ? 6u : 1u;
    unsigned swaps = 3u, g;
    typo_arg ta[4];
    const void *args[1];
    qset q;
    result r;

    ET_SECTION("whole paragraphs: English and its alphabet");
    memset(&ab, 0, sizeof ab);
    if (!corpus_load(&c, "en") || !alphabet_build(&ab, &c)) {
        ET_CHECK(0); engram_free(ab.a); corpus_free(&c); return;
    }
    engram_enc_cfg_default(&cfg);
    mat = index_build(&cfg, ENGRAM_D, &c);
    ET_CHECK(mat != NULL);
    if (!mat) { engram_free(ab.a); corpus_free(&c); return; }
    printf("       %lu paragraphs, alphabet of %lu letters, %s mode\n",
           (unsigned long)c.n, (unsigned long)ab.n, g_quick ? "QUICK" : "FULL");

    ET_SECTION("T3-T5, T10: whole-paragraph typo recall, and separation");
    for (g = 0; g < 4u; g++) {
        static const unsigned ks[] = { 1u, 3u, 5u, 10u };
        ta[g].k = ks[g];
        ta[g].ab = &ab;
        args[0] = &ta[g];
        if (!qset_build(&q, &c, 0u, stride, 1u, v_typos, args, 0x1000u + g) ||
            !measure_dense(&cfg, ENGRAM_D, &c, mat, &q, &r, 1)) { ET_CHECK(0); qset_free(&q); continue; }
        print_result(names[g], &r);
        if (floors[g] > 0.0)
            ET_CHECKF(r.mean >= floors[g], "T3-T5: %s r@1 %.4f < %.2f", names[g], r.mean, floors[g]);
        if (g == 1u) {                                        /* T10 uses the 3-typo source cosine */
            engram_rng rg;
            double s = 0.0, sq = 0.0, mean, sd;
            unsigned k, npairs = 4000u;
            engram_rng_seed(&rg, 0x5E9Au);
            for (k = 0; k < npairs; ) {
                size_t i = (size_t)engram_rng_below(&rg, c.n), j = (size_t)engram_rng_below(&rg, c.n);
                double x;
                if (i == j) continue;                    /* a paragraph is not unrelated to itself */
                x = (double)fdot(mat + i * ENGRAM_D, mat + j * ENGRAM_D, ENGRAM_D);
                s += x; sq += x * x;
                k++;
            }
            mean = s / npairs;
            sd = sqrt(sq / npairs - mean * mean);
            printf("       T10: unrelated pairs mean %.4f sd %.4f | 3-typo source %.4f -> %.1f sd above\n",
                   mean, sd, r.src_cos, (r.src_cos - mean) / sd);
            ET_CHECKF(r.src_cos > mean + 3.0 * sd, "T10: source %.4f not > mean %.4f + 3 sd %.4f",
                      r.src_cos, mean, sd);
        }
        result_free(&r);
        qset_free(&q);
    }

    ET_SECTION("T6: word order -- 3 adjacent swaps");
    args[0] = &swaps;
    if (qset_build(&q, &c, 0u, stride, 1u, v_swaps, args, 0x2003u) &&
        measure_dense(&cfg, ENGRAM_D, &c, mat, &q, &r, 1)) {
        print_result("3 swaps", &r);
        ET_CHECKF(r.mean >= 0.95, "T6: swap r@1 %.4f < 0.95", r.mean);
        result_free(&r);
    } else ET_CHECK(0);
    qset_free(&q);

    engram_free(mat);
    engram_free(ab.a);
    corpus_free(&c);
}

/* ==================================================================================================
 * THE FRAGMENT OBJECTIVE ON ENGLISH: F, T12, T7, T8
 * ============================================================================================== */

/* REGRESSION FLOORS (F). Provenance: this suite at ENGRAM_ENC_ALGO 2, default configuration, the
 * held-out English fragment objective. Each floor is the measurement minus two standard errors
 * clustered by paragraph, rounded DOWN to two places:
 *   FULL   430 paragraphs x 12:  dense 0.8070 (SE 0.0063) -> 0.79   cascade 0.9550 (SE 0.0031) -> 0.94
 *   QUICK   36 paragraphs x 12:  dense 0.8021 (SE 0.0163) -> 0.76   cascade 0.9456 (SE 0.0087) -> 0.92
 * A floor is not a claim made in advance -- it is what was measured, held so it cannot be lost. */
#define FLOOR_DENSE_FULL    0.79
#define FLOOR_CASCADE_FULL  0.94
#define FLOOR_DENSE_QUICK   0.76
#define FLOOR_CASCADE_QUICK 0.92

static void test_fragments(void)
{
    corpus c;
    alphabet ab;
    qset q;
    engram_enc_cfg cfg;
    result base, based, alt, altd, exh;
    size_t stride = g_quick ? 12u : 2u, reversed = 0;
    unsigned k;

    ET_SECTION("fragment objective: English, held-out odd paragraphs, 12 arms");
    memset(&ab, 0, sizeof ab);
    if (!corpus_load(&c, "en") || !alphabet_build(&ab, &c) || !qset_fragments(&q, &c, &ab, 1u, stride)) {
        ET_CHECK(0); engram_free(ab.a); corpus_free(&c); return;
    }
    printf("       %lu query texts from %lu held-out paragraphs, against all %lu\n",
           (unsigned long)q.n, (unsigned long)q.nq, (unsigned long)c.n);
    printf("       %-34s  mean |   5%%   8%%  10%%  15%% | same, 1 typo | same, 2 typos\n", "");
    engram_enc_cfg_default(&cfg);
    if (!run_cascade_config("DEFAULT", &cfg, &c, &q, 0, &base, &based)) goto out0;

    ET_SECTION("saturation guard: both instruments must be able to see an improvement");
    ET_CHECKF(base.mean < SATURATED, "the cascade scores %.4f >= %.2f: this instrument can no longer "
              "fail, and every gate on it would pass vacuously", base.mean, SATURATED);
    ET_CHECKF(based.mean < SATURATED, "the dense stage scores %.4f >= %.2f", based.mean, SATURATED);
    ET_CHECKF(base.refused == 0u, "%lu fragment queries were refused", (unsigned long)base.refused);

    ET_SECTION("F: regression floors");
    {
        double fd = g_quick ? FLOOR_DENSE_QUICK : FLOOR_DENSE_FULL;
        double fc = g_quick ? FLOOR_CASCADE_QUICK : FLOOR_CASCADE_FULL;
        ET_CHECKF(based.mean >= fd, "dense fragment recall %.4f fell below its floor %.3f", based.mean, fd);
        ET_CHECKF(base.mean >= fc, "cascade fragment recall %.4f fell below its floor %.3f", base.mean, fc);
        printf("       floors: dense %.2f, cascade %.2f (%s); measured dense %.4f (SE %.4f), "
               "cascade %.4f (SE %.4f)\n", fd, fc, g_quick ? "QUICK" : "FULL",
               based.mean, clustered_se(&based), base.mean, clustered_se(&base));
    }

    ET_SECTION("T8: the cascade against dense alone, a dense first stage, and exhaustive exact search");
    gate_loses("T8", "dense 512 alone", &base, &based);
    if (run_cascade_config("DEFAULT", &cfg, &c, &q, 1, &alt, NULL)) {
        gate_no_gain("T8", "dense first stage instead of signature", &base, &alt);
        result_free(&alt);
    }
    if (measure_exact_exhaustive(&cfg, &c, &q, &exh, &reversed)) {
        print_result("EXHAUSTIVE exact (reference)", &exh);
        printf("       (%lu paragraphs too long for a table were scored the product's way round)\n",
               (unsigned long)reversed);
        gate_no_gain("T8", "exhaustive exact search", &base, &exh);
        result_free(&exh);
    } else ET_CHECK(0);

    ET_SECTION("T8: no dense-only dimension beats the cascade");
    for (k = 0; k < (g_quick ? 2u : 4u); k++) {
        static const unsigned dims[] = { 256u, 1024u, 2048u, 4096u };
        char name[48];
        snprintf(name, sizeof name, "dense only, D = %u", dims[k]);
        if (!run_config(name, &cfg, dims[k], &c, &q, &alt)) continue;
        gate_no_gain("T8", name, &base, &alt);
        result_free(&alt);
    }

    ET_SECTION("T12: TF shaping of the DENSE stage -- sqrt beats linear; quarter and sign do not beat sqrt");
    for (k = 0; k < 3u; k++) {
        static const engram_tf tfs[] = { ENGRAM_TF_LINEAR, ENGRAM_TF_QUARTER, ENGRAM_TF_SIGN };
        static const char *names[] = { "dense tf = linear", "dense tf = quarter", "dense tf = sign" };
        engram_enc_cfg_default(&cfg);
        cfg.tf = tfs[k];
        if (!run_config(names[k], &cfg, ENGRAM_D, &c, &q, &altd)) continue;
        if (k == 0u) gate_loses("T12", names[k], &based, &altd);
        else gate_no_gain("T12", names[k], &based, &altd);
        result_free(&altd);
    }

    ET_SECTION("T7: ablation -- no single-class change beats the cascade; dropping characters loses");
    for (k = 0; k < 6u; k++) {
        static const char *names[] = { "- char3", "char3 at its former weight 1", "- char4",
                                       "+ words at (1, 1)", "+ words at VECTRA TRACE (4, 6)",
                                       "words only, no characters" };
        engram_enc_cfg_default(&cfg);
        if (k == 0u) cfg.w_char3 = 0.0f;
        if (k == 1u) cfg.w_char3 = 1.0f;
        if (k == 2u) cfg.w_char4 = 0.0f;
        if (k == 3u) { cfg.w_word1 = 1.0f; cfg.w_word2 = 1.0f; }
        if (k == 4u) { cfg.w_word1 = 4.0f; cfg.w_word2 = 6.0f; }
        if (k == 5u) { cfg.w_char3 = cfg.w_char4 = 0.0f; cfg.w_word1 = cfg.w_word2 = 1.0f; }
        if (!run_cascade_config(names[k], &cfg, &c, &q, 0, &alt, &altd)) continue;
        if (k == 5u) gate_loses("T7", names[k], &base, &alt);
        else gate_no_gain("T7", names[k], &base, &alt);
        report_diff("same, dense stage only", &based, &altd);
        result_free(&alt);
        result_free(&altd);
    }

    result_free(&base);
    result_free(&based);
out0:
    qset_free(&q);
    engram_free(ab.a);
    corpus_free(&c);
}

/* ==================================================================================================
 * T13 -- THE SCRIPT SPLIT: Chinese in detail, then every language against VECTRA TRACE's weighting
 * ============================================================================================== */
static void test_script_split(void)
{
    corpus c;
    alphabet ab;
    qset q;
    engram_enc_cfg cfg;
    result base, alt;
    size_t stride = g_quick ? 6u : 2u;
    unsigned k;

    ET_SECTION("T13: Chinese, held-out odd paragraphs -- the ideographic features do the work");
    memset(&ab, 0, sizeof ab);
    if (!corpus_load(&c, "zh") || !alphabet_build(&ab, &c) || !qset_fragments(&q, &c, &ab, 1u, stride)) {
        ET_CHECK(0); engram_free(ab.a); corpus_free(&c); return;
    }
    engram_enc_cfg_default(&cfg);
    if (run_cascade_config("DEFAULT", &cfg, &c, &q, 0, &alt, &base)) {
        /* On Chinese the cascade is near 1 -- it cannot show a gain -- so these gates judge the dense
         * stage, which is not saturated and where the differences are visible. */
        printf("       the Chinese cascade scores %.4f: the gates below judge the dense stage\n", alt.mean);
        result_free(&alt);
        for (k = 0; k < 5u; k++) {
            static const char *names[] = { "- ideographic features", "ideographic weight 4",
                                           "ideographic weight 64", "- character n-grams",
                                           "ordered ideograph pairs" };
            engram_enc_cfg_default(&cfg);
            if (k == 0u) cfg.w_ideo1 = cfg.w_ideo2 = 0.0f;
            if (k == 1u) cfg.w_ideo1 = cfg.w_ideo2 = 4.0f;
            if (k == 2u) cfg.w_ideo1 = cfg.w_ideo2 = 64.0f;
            if (k == 3u) cfg.w_char3 = cfg.w_char4 = 0.0f;
            if (k == 4u) cfg.ordered_pairs = 1;
            if (!run_config(names[k], &cfg, ENGRAM_D, &c, &q, &alt)) continue;
            if (k == 0u) gate_loses("T13", names[k], &base, &alt);
            else gate_no_gain("T13", names[k], &base, &alt);
            result_free(&alt);
        }
        result_free(&base);
    }
    qset_free(&q);
    engram_free(ab.a);
    corpus_free(&c);
}

static void test_languages(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    unsigned li, k;

    ET_SECTION("T11: other languages, whole paragraphs, 3 typos (zh reported: a Chinese typo is a word)");
    for (li = 1; li < 5u; li++) {
        corpus c;
        alphabet ab;
        engram_enc_cfg cfg;
        float *mat;
        typo_arg t3;
        const void *args[1];
        qset q;
        result r;
        char name[48];
        memset(&ab, 0, sizeof ab);
        if (!corpus_load(&c, langs[li]) || !alphabet_build(&ab, &c)) {
            ET_CHECKF(0, "corpus or alphabet for %s did not load", langs[li]);
            engram_free(ab.a); corpus_free(&c); continue;
        }
        engram_enc_cfg_default(&cfg);
        t3.k = 3u; t3.ab = &ab; args[0] = &t3;
        mat = index_build(&cfg, ENGRAM_D, &c);
        if (mat && qset_build(&q, &c, 0u, g_quick ? 3u : 1u, 1u, v_typos, args, 0x7003u) &&
            measure_dense(&cfg, ENGRAM_D, &c, mat, &q, &r, 1)) {
            snprintf(name, sizeof name, "%s: 3 typos, whole", langs[li]);
            print_result(name, &r);
            if (li < 4u) ET_CHECKF(r.mean >= 0.90, "T11: %s r@1 %.4f < 0.90", langs[li], r.mean);
            result_free(&r);
        } else ET_CHECK(0);
        qset_free(&q);
        engram_free(mat);
        engram_free(ab.a);
        corpus_free(&c);
    }

    ET_SECTION("T13: every language, held-out -- the default must beat VECTRA TRACE's weighting");
    printf("       fragment objective, D = 512; dev = even paragraphs (%s), held = odd\n",
           g_quick ? "not run in QUICK mode" : "reported");
    for (li = 0; li < 5u; li++) {
        corpus c;
        alphabet ab;
        qset qe, qo;
        result rdef, rtr;
        size_t stride = g_quick ? 6u : 2u;
        memset(&ab, 0, sizeof ab);
        memset(&qe, 0, sizeof qe);
        memset(&rdef, 0, sizeof rdef);
        memset(&rtr, 0, sizeof rtr);
        if (!corpus_load(&c, langs[li]) || !alphabet_build(&ab, &c) ||
            !qset_fragments(&qo, &c, &ab, 1u, stride) ||
            (!g_quick && !qset_fragments(&qe, &c, &ab, 0u, stride))) {
            ET_CHECKF(0, "query sets for %s did not build", langs[li]);
            engram_free(ab.a); corpus_free(&c); continue;
        }
        for (k = 0; k < 4u; k++) {
            static const char *names[] = { "VECTRA TRACE weights, linear", "VECTRA TRACE weights, sqrt",
                                           "characters only, sqrt", "ENGRAM default" };
            engram_enc_cfg cfg;
            result re, ro;
            float *m;
            if (g_quick && k != 0u && k != 3u) continue;
            if (k <= 1u) cfg_trace(&cfg); else engram_enc_cfg_default(&cfg);
            if (k == 1u) cfg.tf = ENGRAM_TF_SQRT;
            if (k == 2u) cfg.w_ideo1 = cfg.w_ideo2 = 0.0f;
            memset(&re, 0, sizeof re);
            m = index_build(&cfg, ENGRAM_D, &c);
            if (!m || !measure_dense(&cfg, ENGRAM_D, &c, m, &qo, &ro, 0) ||
                (!g_quick && !measure_dense(&cfg, ENGRAM_D, &c, m, &qe, &re, 0))) {
                ET_CHECK(0); engram_free(m); result_free(&re); continue;
            }
            engram_free(m);
            if (g_quick) printf("       %s  %-30s held %.4f\n", langs[li], names[k], ro.mean);
            else printf("       %s  %-30s dev %.4f  held %.4f\n", langs[li], names[k], re.mean, ro.mean);
            result_free(&re);
            if (k == 0u) rtr = ro;                        /* ownership of ro.hit moves */
            else if (k == 3u) rdef = ro;
            else result_free(&ro);
        }
        {                                   /* the shipped retrieval on the same held-out queries */
            engram_enc_cfg cfg;
            float *m;
            uint64_t *sg;
            result rc;
            engram_enc_cfg_default(&cfg);
            m = index_build(&cfg, ENGRAM_D, &c);
            sg = sig_build(&cfg, &c);
            if (m && sg && measure_cascade(&cfg, &c, m, sg, &qo, CASCADE_C, &rc, NULL)) {
                printf("       %s  %-30s held %.4f\n", langs[li], "ENGRAM default, cascade", rc.mean);
                result_free(&rc);
            } else ET_CHECK(0);
            engram_free(m);
            engram_free(sg);
        }
        if (rdef.hit && rtr.hit) {
            paired p = compare(&rtr, &rdef);
            printf("         %s: default over VECTRA TRACE weighting %+.4f (margin %.4f)  must win\n",
                   langs[li], p.diff, p.margin);
            ET_CHECKF(p.diff > p.margin, "T13: %s -- the default does not beat VECTRA TRACE's weighting "
                      "beyond the margin (%+.4f, margin %.4f)", langs[li], p.diff, p.margin);
        } else ET_CHECK(0);
        result_free(&rdef);
        result_free(&rtr);
        qset_free(&qe);
        qset_free(&qo);
        engram_free(ab.a);
        corpus_free(&c);
    }
}

/* ==================================================================================================
 * T9 -- FOLDING
 * ============================================================================================== */
static void test_folding_proof(void)
{
    corpus c;
    engram_enc_cfg cfg;
    float *mat, v[ENGRAM_D], w[ENGRAM_D];
    char *norm;
    size_t i, nl;
    unsigned identical = 0, total = 0;
    const void *args[1] = { NULL };
    qset q;
    result compat, casectl;

    ET_SECTION("T9: folding -- idempotent under COMPAT, and a CASE-only control loses similarity");
    ET_CHECK(corpus_load(&c, "fr"));
    if (!c.n) { corpus_free(&c); return; }
    for (i = 0; i < c.n; i++) {
        if (engram_text_normalize(c.line[i], c.len[i], ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT,
                                  &norm, &nl) != ENGRAM_OK) continue;
        if (engram_encode(NULL, c.line[i], c.len[i], v, NULL) == ENGRAM_OK &&
            engram_encode(NULL, norm, nl, w, NULL) == ENGRAM_OK && memcmp(v, w, sizeof v) == 0)
            identical++;
        total++;
        engram_free(norm);
    }
    printf("       %u of %u French paragraphs encode bit-identically to their unaccented lowercase form\n",
           identical, total);
    ET_EQ_U64(identical, total);

    if (!qset_build(&q, &c, 0u, 1u, 1u, v_normalised, args, 0x9001u)) {
        ET_CHECK(0); qset_free(&q); corpus_free(&c); return;
    }
    engram_enc_cfg_default(&cfg);
    mat = index_build(&cfg, ENGRAM_D, &c);
    if (mat && measure_dense(&cfg, ENGRAM_D, &c, mat, &q, &compat, 1)) {
        engram_free(mat);
        print_result("unaccented queries, COMPAT (default)", &compat);
        cfg.fold = ENGRAM_FOLD_CASE;
        mat = index_build(&cfg, ENGRAM_D, &c);
        if (mat && measure_dense(&cfg, ENGRAM_D, &c, mat, &q, &casectl, 1)) {
            print_result("unaccented queries, CASE only (control)", &casectl);
            ET_CHECK(compat.mean == 1.0);
            ET_CHECKF(casectl.src_cos < compat.src_cos - 0.01,
                      "the control's source cosine %.4f is not below COMPAT's %.4f -- folding is not "
                      "doing the work", casectl.src_cos, compat.src_cos);
            result_free(&casectl);
        } else ET_CHECK(0);
        result_free(&compat);
    } else ET_CHECK(0);
    engram_free(mat);
    qset_free(&q);
    corpus_free(&c);
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = (q && q[0] == '1');
    printf("ENGRAM P1.2 -- encoder\n");
    ET_SELFTEST();
    test_contract();
    test_exact();
    test_signature();
    test_fingerprint();
    test_whole();
    test_fragments();
    test_script_split();
    test_languages();
    test_folding_proof();
    return et_report("test_enc");
}
