/* ==================================================================================================
 * engram_enc.c -- the streaming feature-hashed encoder. Design and guarantees in engram_enc.h.
 * ============================================================================================== */
#include "engram_enc.h"

#include <math.h>
#include <string.h>

/* ==================================================================================================
 * FEATURE IDENTITIES
 * ==================================================================================================
 * Every feature is a 64-bit id f. The dense sink turns f into a dimension and a sign; the exact sink
 * uses f itself as the key. Distinct tags put the feature classes in disjoint families: the character
 * trigram "the" and the word "the" are different features and must not share an id.
 *
 * COST AND COLLISIONS, both argued rather than hoped:
 *   char3   f3 = mix64(P3 ^ pack3(a,b,c)). pack3 is injective (three 21-bit codepoints in 63 bits)
 *           and mix64 is a bijection, so two distinct trigrams NEVER share an id.
 *   char4   f4 = mix64(f3' ^ P4 ^ d*K4), where f3' is the id of the trigram (a,b,c) computed one
 *           character earlier -- the 4-gram costs one mix, not four. For the same (a,b,c) the map
 *           d -> d*K4 is injective (K4 odd), so ids differ; for different trigrams f3' are distinct
 *           outputs of a bijection, and a collision needs f3' ^ f3'' to equal a specific value:
 *           probability 2^-64 per pair.
 *   word    FNV-1a over the codepoints, finalised with the length; unigram and bigram ids mix that
 *           with their own family seed. Random 2^-64 collisions, once per word, not per character.
 *   dense   h = mix64(f ^ K_DIM): the dimension is the HIGH 32 bits scaled into [0, d) by
 *           multiply-shift (Lemire, 2019), the sign is the low bit. One mix, no division. The bias of
 *           multiply-shift for non-power-of-two d is at most d / 2^32 -- below 1e-6 for any legal d.
 * The per-family seeds P = mix64(TAG ^ seed) are computed once per text, not per feature. */
#define TAG_CHAR3 0x6368617233A5A501ull
#define TAG_CHAR4 0x6368617234A5A502ull
#define TAG_WORD1 0x776F726431A5A503ull
#define TAG_WORD2 0x776F726432A5A504ull
#define K_CHAR4   0x9E3779B97F4A7C15ull           /* odd */
#define K_DIM     0xD1B54A32D192ED03ull
#define K_SIG     0xA0761D6478BD642Full

ENGRAM_STATIC_ASSERT(ENGRAM_SIG_K == 2u, encq_sbit_holds_two_bits_per_feature);
ENGRAM_STATIC_ASSERT(ENGRAM_SIG_LOG2 * ENGRAM_SIG_K <= 64u, sig_bits_fit_one_hash);
ENGRAM_STATIC_ASSERT(ENGRAM_ENCQ_SLOTS <= 65536u, encq_occ_fits_16_bits);
ENGRAM_STATIC_ASSERT((ENGRAM_ENCQ_SLOTS & (ENGRAM_ENCQ_SLOTS - 1u)) == 0u, encq_slots_power_of_two);
#define WORD_BASIS 0xCBF29CE484222325ull
#define WORD_PRIME 0x00000100000001B3ull

void engram_enc_cfg_default(engram_enc_cfg *cfg)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof *cfg);
    cfg->w_char3 = 0.0625f;              /* 1/16: exact in binary, so sums stay exact */
    cfg->w_char4 = 1.0f;
    cfg->w_word1 = 0.0f;
    cfg->w_word2 = 0.0f;
    cfg->w_ideo1 = 16.0f;
    cfg->w_ideo2 = 16.0f;
    cfg->ordered_pairs = 0;
    cfg->fold = ENGRAM_FOLD_COMPAT;
    cfg->utf8 = ENGRAM_UTF8_REPLACE;
    cfg->seed = ENGRAM_ENC_SEED;
    cfg->tf = ENGRAM_TF_SQRT;
}

/* Finite and non-negative, tested on the BIT PATTERN: a comparison like `w >= 0.0f` is false for NaN,
 * which would do, but -ffast-math is entitled to assume NaN cannot occur and fold it away. The bits
 * cannot be optimised into a lie. */
static int engram_weight_ok(float w)
{
    uint32_t u;
    memcpy(&u, &w, sizeof u);
    if ((u & 0x7F800000u) == 0x7F800000u) return 0;      /* NaN or infinity */
    if ((u & 0x80000000u) && (u & 0x7FFFFFFFu)) return 0; /* negative, other than -0.0 */
    return 1;
}

engram_rc engram_enc_cfg_check(const engram_enc_cfg *cfg)
{
    if (!cfg) return ENGRAM_E_ARG;
    if (!engram_weight_ok(cfg->w_char3) || !engram_weight_ok(cfg->w_char4) ||
        !engram_weight_ok(cfg->w_word1) || !engram_weight_ok(cfg->w_word2) ||
        !engram_weight_ok(cfg->w_ideo1) || !engram_weight_ok(cfg->w_ideo2)) return ENGRAM_E_ARG;
    if (!(cfg->w_char3 > 0.0f || cfg->w_char4 > 0.0f || cfg->w_word1 > 0.0f || cfg->w_word2 > 0.0f ||
          cfg->w_ideo1 > 0.0f || cfg->w_ideo2 > 0.0f))
        return ENGRAM_E_ARG;
    if (cfg->fold != ENGRAM_FOLD_NONE && cfg->fold != ENGRAM_FOLD_CASE && cfg->fold != ENGRAM_FOLD_COMPAT)
        return ENGRAM_E_ARG;
    if (cfg->utf8 != ENGRAM_UTF8_STRICT && cfg->utf8 != ENGRAM_UTF8_REPLACE) return ENGRAM_E_ARG;
    if (cfg->tf != ENGRAM_TF_LINEAR && cfg->tf != ENGRAM_TF_SQRT && cfg->tf != ENGRAM_TF_QUARTER &&
        cfg->tf != ENGRAM_TF_SIGN) return ENGRAM_E_ARG;
    return ENGRAM_OK;
}

/* ==================================================================================================
 * THE STREAMING STATE
 * ==================================================================================================
 * ONE feature generator feeds three sinks, so the dense vector, the exact query table and the exact
 * document scan see the same features BY CONSTRUCTION (rule R7) -- not by two implementations that
 * are hoped to agree. */
typedef enum { SINK_DENSE = 0, SINK_QBUILD = 1, SINK_DSCAN = 2, SINK_SIG = 3 } engram_sink;

/* Signature bit i of feature f: the i-th ENGRAM_SIG_LOG2-bit field from the top of one mix. */
static unsigned engram_sig_bit(uint64_t f, unsigned i)
{
    uint64_t h = engram_mix64(f ^ K_SIG);
    return (unsigned)((h >> (64u - ENGRAM_SIG_LOG2 * (i + 1u))) & (ENGRAM_SIG_BITS - 1u));
}

typedef struct {
    engram_sink       sink;
    engram_encq      *q;            /* QBUILD and DSCAN */
    double            dmass;        /* DSCAN: the document's total feature weight */
    uint64_t         *sig;          /* SIG */
    int               full;         /* QBUILD: the table ran out of room */
    double           *acc;
    unsigned          d;
    uint64_t          p3, p4, p1, p2;  /* per-family seeds, mix64(TAG ^ seed) */
    uint64_t          f3prev;          /* id of the trigram that ended at the previous character */
    double            w3, w4, w1, w2, wi1, wi2;
    int               ordered;
    uint32_t          win[3];       /* the last three stream codepoints; win[2] the most recent */
    uint64_t          nstream;
    int               last_space;
    int               in_word;
    int               word_ideo;    /* the word being built is a single ideograph */
    uint64_t          wh, wlen;
    uint64_t          prev;
    int               have_prev;
    int               prev_ideo;
    engram_enc_stats *st;
} engram_encst;

/* The exact tables key on the 64-bit feature hash; 0 marks an empty slot, so a feature that hashes to
 * 0 is stored as 1. Two distinct features then share a key with probability 2^-64 per pair. */
static uint64_t engram_qkey(uint64_t f) { return f ? f : 1u; }

static void engram_q_insert(engram_encst *s, uint64_t f, double w)
{
    engram_encq *q = s->q;
    uint64_t k = engram_qkey(f);
    unsigned i = (unsigned)(k & (ENGRAM_ENCQ_SLOTS - 1u));
    for (;;) {
        if (q->key[i] == k) { q->qw[i] += w; break; }
        if (q->key[i] == 0u) {
            if (q->n >= ENGRAM_ENCQ_MAXF) { s->full = 1; return; }
            unsigned b;
            q->key[i] = k;
            q->qw[i] = w;
            for (b = 0; b < ENGRAM_SIG_K; b++) q->sbit[q->n][b] = (uint16_t)engram_sig_bit(f, b);
            q->occ[q->n++] = (uint16_t)i;
            break;
        }
        i = (i + 1u) & (ENGRAM_ENCQ_SLOTS - 1u);
    }
    q->qmass += w;
}

static void engram_q_probe(engram_encst *s, uint64_t f, double w)
{
    engram_encq *q = s->q;
    uint64_t k = engram_qkey(f);
    unsigned i = (unsigned)(k & (ENGRAM_ENCQ_SLOTS - 1u));
    s->dmass += w;
    for (;;) {                                 /* terminates: the table is at most half full */
        if (q->key[i] == k) { q->dw[i] += w; return; }
        if (q->key[i] == 0u) return;
        i = (i + 1u) & (ENGRAM_ENCQ_SLOTS - 1u);
    }
}

static void engram_add_feature(engram_encst *s, uint64_t f, double w)
{
    if (s->sink == SINK_DENSE) {
        uint64_t h = engram_mix64(f ^ K_DIM);
        unsigned dim = (unsigned)(((h >> 32) * (uint64_t)s->d) >> 32);
        s->acc[dim] += (h & 1u) ? -w : w;
    } else if (s->sink == SINK_QBUILD) {
        engram_q_insert(s, f, w);
    } else if (s->sink == SINK_DSCAN) {
        engram_q_probe(s, f, w);
    } else {
        unsigned i;
        (void)w;
        for (i = 0; i < ENGRAM_SIG_K; i++) {
            unsigned b = engram_sig_bit(f, i);
            s->sig[b >> 6] |= (uint64_t)1u << (b & 63u);
        }
        s->dmass += 1.0;                        /* counts features, so E_SHORT can be told */
    }
}

/* Codepoints are at most 21 bits, so three of them pack losslessly into 63: the trigram key is the
 * trigram itself, with no hashing and so no possibility of two trigrams sharing a key. */
static uint64_t engram_pack3(uint32_t a, uint32_t b, uint32_t c)
{
    return (uint64_t)a | ((uint64_t)b << 21) | ((uint64_t)c << 42);
}

static void engram_stream_push(engram_encst *s, uint32_t cp)
{
    if (s->nstream >= 3u && s->w4 > 0.0) {       /* uses the PREVIOUS trigram's id: compute first */
        engram_add_feature(s, engram_mix64(s->f3prev ^ s->p4 ^ ((uint64_t)cp * K_CHAR4)), s->w4);
        s->st->f_char4++;
    }
    if (s->nstream >= 2u) {
        uint64_t f3 = engram_mix64(s->p3 ^ engram_pack3(s->win[1], s->win[2], cp));
        if (s->w3 > 0.0) {
            engram_add_feature(s, f3, s->w3);
            s->st->f_char3++;
        }
        s->f3prev = f3;
    }
    s->win[0] = s->win[1];
    s->win[1] = s->win[2];
    s->win[2] = cp;
    s->nstream++;
    s->last_space = (cp == 0x20u);
}

static void engram_stream_space(engram_encst *s)
{
    if (!s->last_space) engram_stream_push(s, 0x20u);
}

static void engram_word_add(engram_encst *s, uint32_t cp, int ideo)
{
    if (!s->in_word) {
        s->in_word = 1;
        s->word_ideo = ideo;
        s->wh = WORD_BASIS;
        s->wlen = 0;
    }
    s->wh = (s->wh ^ (uint64_t)cp) * WORD_PRIME;
    s->wlen++;
}

/* The unigram and bigram weights depend on the script of the words involved; the hash families do
 * not, because a word's key already contains its codepoints. */
static void engram_word_end(engram_encst *s)
{
    uint64_t w;
    int ideo;
    double wu, wb;
    if (!s->in_word) return;
    s->in_word = 0;
    ideo = s->word_ideo;
    w = engram_mix2(s->wh, s->wlen);
    s->st->words++;
    if (ideo) s->st->ideographs++;
    wu = ideo ? s->wi1 : s->w1;
    if (wu > 0.0) {
        engram_add_feature(s, engram_mix64(s->p1 ^ w), wu);
        if (ideo) s->st->f_ideo1++; else s->st->f_word1++;
    }
    wb = (ideo && s->prev_ideo) ? s->wi2 : s->w2;
    if (s->have_prev && wb > 0.0) {
        uint64_t a = s->prev, b = w;
        if (!s->ordered && a > b) { uint64_t t = a; a = b; b = t; }
        engram_add_feature(s, engram_mix2(s->p2 ^ a, b), wb);
        if (ideo && s->prev_ideo) s->st->f_ideo2++; else s->st->f_word2++;
    }
    s->prev = w;
    s->prev_ideo = ideo;
    s->have_prev = 1;
}

/* Set up the generator for cfg (already checked) and text. */
static engram_rc engram_walk_init(engram_encst *s, engram_textit *it, const engram_enc_cfg *cfg,
                                  const void *text, size_t n, engram_enc_stats *st)
{
    memset(s, 0, sizeof *s);
    s->p3 = engram_mix64(TAG_CHAR3 ^ cfg->seed);
    s->p4 = engram_mix64(TAG_CHAR4 ^ cfg->seed);
    s->p1 = engram_mix64(TAG_WORD1 ^ cfg->seed);
    s->p2 = engram_mix64(TAG_WORD2 ^ cfg->seed);
    s->w3 = (double)cfg->w_char3;
    s->w4 = (double)cfg->w_char4;
    s->w1 = (double)cfg->w_word1;
    s->w2 = (double)cfg->w_word2;
    s->wi1 = (double)cfg->w_ideo1;
    s->wi2 = (double)cfg->w_ideo2;
    s->ordered = cfg->ordered_pairs ? 1 : 0;
    s->st = st;
    st->bytes = n;
    return engram_textit_init(it, text, n, cfg->fold, cfg->utf8);
}

/* Run the whole text through the generator into whichever sink s is set to. */
static engram_rc engram_walk(engram_encst *s, engram_textit *it)
{
    uint32_t cp;
    unsigned k;
    engram_stream_push(s, 0x20u);                  /* the leading word boundary */
    while (engram_textit_next(it, &cp, &k)) {
        if (k == ENGRAM_CLASS_SPACE) {
            engram_word_end(s);
            engram_stream_space(s);
        } else if (k == ENGRAM_CLASS_PUNCT) {
            engram_word_end(s);
            engram_stream_push(s, cp);
        } else if (k == ENGRAM_CLASS_IDEO) {
            engram_word_end(s);
            engram_word_add(s, cp, 1);
            engram_word_end(s);
            engram_stream_push(s, cp);
        } else {                                   /* WORD and MARK */
            engram_word_add(s, cp, 0);
            engram_stream_push(s, cp);
        }
    }
    s->st->codepoints   = it->n_codepoints;
    s->st->invalid_utf8 = it->n_invalid;
    s->st->dropped      = it->n_dropped;
    if (it->err != ENGRAM_OK) return it->err;
    engram_word_end(s);
    engram_stream_space(s);                        /* the trailing word boundary */
    return ENGRAM_OK;
}

/* ==================================================================================================
 * ENCODE
 * ============================================================================================== */
engram_rc engram_encode_d(const engram_enc_cfg *cfg, const void *text, size_t n,
                          float *out, unsigned d, engram_enc_stats *st)
{
    double acc[ENGRAM_ENC_DMAX];
    engram_enc_cfg defcfg;
    engram_enc_stats local;
    engram_encst s;
    engram_textit it;
    unsigned i;
    double sumsq = 0.0, norm;
    engram_rc rc;

    if (!st) st = &local;
    memset(st, 0, sizeof *st);
    if (!out || d < 64u || d > ENGRAM_ENC_DMAX || (d % 8u) != 0u) return ENGRAM_E_ARG;
    memset(out, 0, (size_t)d * sizeof *out);
    if (!cfg) { engram_enc_cfg_default(&defcfg); cfg = &defcfg; }
    rc = engram_enc_cfg_check(cfg);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_walk_init(&s, &it, cfg, text, n, st);
    if (rc != ENGRAM_OK) return rc;
    memset(acc, 0, (size_t)d * sizeof acc[0]);
    s.sink = SINK_DENSE;
    s.acc = acc;
    s.d = d;
    rc = engram_walk(&s, &it);
    if (rc != ENGRAM_OK) return rc;                 /* out is still all zeros */

    /* TF shaping. The accumulator holds exact integers, and sqrt is correctly rounded, so the shaped
     * vector is the same bits on every IEEE-754 machine. */
    if (cfg->tf != ENGRAM_TF_LINEAR) {
        for (i = 0; i < d; i++) {
            double x = acc[i], a = x < 0.0 ? -x : x;
            if (cfg->tf == ENGRAM_TF_SIGN) a = a > 0.0 ? 1.0 : 0.0;
            else {
                a = sqrt(a);
                if (cfg->tf == ENGRAM_TF_QUARTER) a = sqrt(a);
            }
            acc[i] = x < 0.0 ? -a : a;
        }
    }
    for (i = 0; i < d; i++) sumsq += acc[i] * acc[i];
    norm = sqrt(sumsq);
    st->norm = norm;
    if (!(norm > 0.0)) return ENGRAM_E_SHORT;
    for (i = 0; i < d; i++) out[i] = (float)(acc[i] / norm);
    return ENGRAM_OK;
}

engram_rc engram_encode(const engram_enc_cfg *cfg, const void *text, size_t n,
                        float out[ENGRAM_D], engram_enc_stats *st)
{
    return engram_encode_d(cfg, text, n, out, ENGRAM_D, st);
}

/* ==================================================================================================
 * THE EXACT STAGE
 * ============================================================================================== */
engram_rc engram_encq_build(engram_encq *q, const engram_enc_cfg *cfg, const void *text, size_t n)
{
    engram_enc_stats st;
    engram_encst s;
    engram_textit it;
    engram_rc rc;
    if (!q) return ENGRAM_E_ARG;
    memset(q, 0, sizeof *q);                        /* not ready, every key empty */
    if (cfg) q->cfg = *cfg; else engram_enc_cfg_default(&q->cfg);
    rc = engram_enc_cfg_check(&q->cfg);
    if (rc != ENGRAM_OK) return rc;
    if (q->cfg.tf != ENGRAM_TF_SQRT) return ENGRAM_E_ARG;
    rc = engram_walk_init(&s, &it, &q->cfg, text, n, &st);
    if (rc != ENGRAM_OK) return rc;
    s.sink = SINK_QBUILD;
    s.q = q;
    rc = engram_walk(&s, &it);
    if (rc != ENGRAM_OK) return rc;
    if (s.full) return ENGRAM_E_FULL;
    if (!(q->qmass > 0.0)) return ENGRAM_E_SHORT;
    q->ready = 1;
    return ENGRAM_OK;
}

engram_rc engram_encq_scores(engram_encq *q, const void *text, size_t n, double *exact, double *contain)
{
    engram_enc_stats st;
    engram_encst s;
    engram_textit it;
    engram_rc rc;
    double num = 0.0, held = 0.0;
    unsigned j;
    if (exact) *exact = 0.0;
    if (contain) *contain = 0.0;
    if (!q) return ENGRAM_E_ARG;
    if (!q->ready) return ENGRAM_E_STATE;
    for (j = 0; j < q->n; j++) q->dw[q->occ[j]] = 0.0;
    rc = engram_walk_init(&s, &it, &q->cfg, text, n, &st);
    if (rc != ENGRAM_OK) return rc;
    s.sink = SINK_DSCAN;
    s.q = q;
    rc = engram_walk(&s, &it);
    if (rc != ENGRAM_OK) return rc;
    if (!(s.dmass > 0.0)) return ENGRAM_E_SHORT;
    for (j = 0; j < q->n; j++) {
        unsigned i = q->occ[j];
        if (q->dw[i] > 0.0) { num += sqrt(q->qw[i] * q->dw[i]); held += q->qw[i]; }
    }
    if (exact) {
        *exact = num / sqrt(q->qmass * s.dmass);
        if (*exact > 1.0) *exact = 1.0;              /* only rounding can put it there */
    }
    if (contain) {
        *contain = held / q->qmass;
        if (*contain > 1.0) *contain = 1.0;          /* the same */
    }
    return ENGRAM_OK;
}

engram_rc engram_encq_score(engram_encq *q, const void *text, size_t n, double *score)
{
    if (score) *score = 0.0;
    if (!q || !score) return ENGRAM_E_ARG;
    return engram_encq_scores(q, text, n, score, NULL);
}

/* ==================================================================================================
 * THE SIGNATURE
 * ============================================================================================== */
engram_rc engram_encode_sig(const engram_enc_cfg *cfg, const void *text, size_t n,
                            uint64_t sig[ENGRAM_SIG_WORDS], engram_enc_stats *st)
{
    engram_enc_cfg defcfg;
    engram_enc_stats local;
    engram_encst s;
    engram_textit it;
    engram_rc rc;
    if (!st) st = &local;
    memset(st, 0, sizeof *st);
    if (!sig) return ENGRAM_E_ARG;
    memset(sig, 0, ENGRAM_SIG_WORDS * sizeof *sig);
    if (!cfg) { engram_enc_cfg_default(&defcfg); cfg = &defcfg; }
    rc = engram_enc_cfg_check(cfg);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_walk_init(&s, &it, cfg, text, n, st);
    if (rc != ENGRAM_OK) return rc;
    s.sink = SINK_SIG;
    s.sig = sig;
    rc = engram_walk(&s, &it);
    if (rc != ENGRAM_OK) {
        memset(sig, 0, ENGRAM_SIG_WORDS * sizeof *sig);      /* a partial signature is not stored */
        return rc;
    }
    return s.dmass > 0.0 ? ENGRAM_OK : ENGRAM_E_SHORT;
}

engram_rc engram_encq_sig_score(const engram_encq *q, const uint64_t sig[ENGRAM_SIG_WORDS],
                                double *score)
{
    double found = 0.0;
    unsigned j, b;
    if (score) *score = 0.0;
    if (!q || !sig || !score) return ENGRAM_E_ARG;
    if (!q->ready) return ENGRAM_E_STATE;
    for (j = 0; j < q->n; j++) {
        int all = 1;
        for (b = 0; b < ENGRAM_SIG_K && all; b++) {
            unsigned bit = q->sbit[j][b];
            if (!((sig[bit >> 6] >> (bit & 63u)) & 1u)) all = 0;
        }
        if (all) found += q->qw[q->occ[j]];
    }
    *score = found / q->qmass;
    return ENGRAM_OK;
}

uint64_t engram_enc_geometry(const engram_enc_cfg *cfg, unsigned d)
{
    engram_enc_cfg c;
    uint32_t w[6];
    const char *uv = engram_unicode_version();
    uint64_t h;
    if (cfg) c = *cfg; else engram_enc_cfg_default(&c);
    memcpy(&w[0], &c.w_char3, 4u);
    memcpy(&w[1], &c.w_char4, 4u);
    memcpy(&w[2], &c.w_word1, 4u);
    memcpy(&w[3], &c.w_word2, 4u);
    memcpy(&w[4], &c.w_ideo1, 4u);
    memcpy(&w[5], &c.w_ideo2, 4u);
    h = engram_hash_bytes("ENGRAM-ENC", 10u, ENGRAM_ENC_ALGO);
    h = engram_mix2(h, (uint64_t)d);
    h = engram_mix2(h, c.seed);
    h = engram_mix2(h, ((uint64_t)w[0] << 32) | w[1]);
    h = engram_mix2(h, ((uint64_t)w[2] << 32) | w[3]);
    h = engram_mix2(h, ((uint64_t)w[4] << 32) | w[5]);
    h = engram_mix2(h, (uint64_t)(c.ordered_pairs ? 1u : 0u));
    h = engram_mix2(h, (uint64_t)c.fold);
    h = engram_mix2(h, (uint64_t)c.tf);
    h = engram_mix2(h, ((uint64_t)ENGRAM_SIG_BITS << 8) | ENGRAM_SIG_K);   /* stored signatures too */
    h = engram_mix2(h, engram_hash_bytes(uv, strlen(uv), 0u));
    return h;
}

/* ==================================================================================================
 * VECTOR ARITHMETIC
 * ============================================================================================== */
float engram_vec_dot(const float *a, const float *b, unsigned d)
{
    double s = 0.0;
    unsigned i;
    if (!a || !b) return 0.0f;
    for (i = 0; i < d; i++) s += (double)a[i] * (double)b[i];
    return (float)s;
}

double engram_vec_norm(const float *a, unsigned d)
{
    double s = 0.0;
    unsigned i;
    if (!a) return 0.0;
    for (i = 0; i < d; i++) s += (double)a[i] * (double)a[i];
    return sqrt(s);
}

engram_rc engram_vec_normalize(float *a, unsigned d)
{
    double n;
    unsigned i;
    if (!a || !d) return ENGRAM_E_ARG;
    n = engram_vec_norm(a, d);
    if (!(n > 0.0)) return ENGRAM_E_SHORT;
    for (i = 0; i < d; i++) a[i] = (float)((double)a[i] / n);
    return ENGRAM_OK;
}
