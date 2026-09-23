/* ==================================================================================================
 * engram_enc.h -- text to its three representations: dense vector, signature, exact score.
 * ==================================================================================================
 *
 * THE PROBLEM
 * ==================================================================================================
 * A person remembers what they wrote approximately: misspelled, reordered, half of it. Every exact
 * matcher on their machine returns nothing for "posgtres conection timout", because no file contains
 * that string; the paragraph they want says "postgres connection timeout" and is four edits away.
 *
 * THREE REPRESENTATIONS, ONE FEATURE GENERATOR
 * ==================================================================================================
 * Every text is read by ONE streaming feature generator, and what the features become depends on the
 * question being asked:
 *
 *   DENSE VECTOR     512 floats, 2 KB     "how similar are these two texts, overall?"
 *                    the geometry the slow store learns in and the router partitions (Phase 2+)
 *   SIGNATURE        8192 bits, 1 KB      "how much of this query could that memory contain?"
 *                    the FIRST stage of retrieval: finds the candidates among everything
 *   EXACT SCORE      nothing stored        "how much, exactly?" -- the Bhattacharyya coefficient
 *                    the SECOND stage: streams each candidate's text and decides
 *
 * Because all three are built by the same generator, a query and a memory can never be described by
 * two different functions (rule R7), and the dense vector is a sketch of exactly the kernel the exact
 * stage computes: with no hash collisions the two are equal.
 *
 * THE FEATURES, WITH THE n THE SCRIPT CALLS FOR
 * ==================================================================================================
 *   ALPHABETIC scripts (Latin, Cyrillic, Greek, Hangul, Indic, ...): a letter says almost nothing,
 *   so the features are character 4-grams (and, faintly, 3-grams) over the normalised stream,
 *   spaces included. A typo destroys the handful of n-grams that cross it and leaves the rest --
 *   which is why character n-grams out-retrieve whole words for European languages (McNamee &
 *   Mayfield, Information Retrieval 7, 2004).
 *
 *   IDEOGRAPHIC scripts (Han, Hiragana, Katakana): one character is roughly one morpheme, so the
 *   features are character unigrams and ADJACENT-PAIR bigrams -- the overlapping-bigram practice of
 *   CJK retrieval. A 3- or 4-gram of ideographs is a rare phrase, and one wrong character destroys
 *   seven of them.
 *
 * Every family is always computed; the WEIGHTS select. Defaults, and the measurement behind each:
 *
 *      character 4-grams             1
 *      character 3-grams             1/16  a FLOOR: the only feature a one-letter text has. Measured,
 *                                          smaller is better all the way to 0 -- but 0 would leave
 *                                          "R" or "C" with nothing to encode, so the smallest tested
 *                                          positive weight (exact in binary, so sums stay exact)
 *      alphabetic word uni/bigrams   0     measured to add nothing once TF is shaped
 *      ideograph uni/bigrams         16    measured flat from 8 to 64; 16 is the middle
 *      term frequency                signed square root (below)
 *
 * THE MEASUREMENT (test_enc.c re-runs every arm; ledger rows W-P1.2-5 to -12 record each turn)
 * ==================================================================================================
 * The task that decides is the one people actually have: a FRAGMENT of a paragraph -- 5% to 15% of
 * it, 0 to 2 typos -- against every paragraph of the corpus. Whole-paragraph queries were abandoned
 * as a selection task because every candidate scored 1.0 on them: an instrument that cannot fail
 * cannot choose (rule R6). Weights were chosen on a DEVELOPMENT split (even paragraphs as queries)
 * and confirmed on the HELD-OUT odd paragraphs. Mean fragment recall@1, held-out:
 *
 *                                          en       fr       de       sv       zh
 *      VECTRA TRACE weights, linear, dense  0.592    0.655    0.692    0.712    0.918
 *      same weights, sqrt, dense            0.705    0.764    0.778    0.791    0.953
 *      characters only, sqrt, dense         0.807    0.865    0.868    0.880    0.752   <- Chinese collapses
 *      ENGRAM default, dense                0.807    0.865    0.868    0.880    0.965
 *      ENGRAM default, CASCADE              0.955    0.980    0.971    0.977    0.998   <- what a person gets
 *
 * Words did not earn their place in alphabetic text; characters did not earn theirs in Chinese. The
 * split by script is the only configuration on the best plateau for all five. On English, where
 * the suite also runs an exhaustive exact search of the whole corpus as the reference, the cascade
 * -- a signature to find, the exact score to decide -- scores above it: 0.955 against 0.952.
 *
 * WHAT IS DIFFERENT FROM VECTRA TRACE, AND WHY
 * ==================================================================================================
 *   CODEPOINTS, NOT BYTES. TRACE hashed byte n-grams, so a "3-gram" of Chinese was one character and
 *   of French was sometimes half of one. Here the unit is the codepoint, after Unicode folding.
 *
 *   FOLDING. "Ecole" and its accented spelling, fullwidth and ASCII, sharp-s and "ss", are the same
 *   features (engram_text.h), so a person who types without accents finds what they wrote with them.
 *
 *   SCRIPT-AWARE. TRACE applied one weighting to every language. Measured, no single weighting is on
 *   the best plateau for both alphabetic and ideographic text; see above.
 *
 *   STREAMING. The encoder holds a three-codepoint window and one running word hash. It allocates
 *   NOTHING, so it cannot fail for memory, and a ten-megabyte document costs the same memory as a
 *   ten-byte one.
 *
 *   EXACT. Every default weight is a small integer or a power-of-two fraction, and features
 *   accumulate in double, so the pre-shaping vector is exact dyadic arithmetic -- the same bits in any summation order on any
 *   IEEE-754 machine. The shaping and the normalisation use only sqrt and division, both correctly
 *   rounded, so the final vector is the same bits everywhere too (rule R5; test_enc's FINGERPRINT).
 *
 * WHAT IT IS NOT
 * ==================================================================================================
 * LEXICAL, NOT SEMANTIC. It matches text that shares characters and words. "car" does not find
 * "automobile". No part of ENGRAM claims otherwise about this layer: meaning, where there is any, is
 * what the slow store learns (Phase 2), and it is measured there, not assumed here.
 * ============================================================================================== */
#ifndef ENGRAM_ENC_H
#define ENGRAM_ENC_H

#include "engram.h"
#include "engram_text.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The largest dimension engram_encode_d accepts. The accumulator lives on the stack, so this bounds
 * it: 4096 doubles, 32 KB -- comfortably inside any thread's stack on either platform. */
#define ENGRAM_ENC_DMAX 4096u

/* Bumped whenever the DEFINITION of a feature changes, so geometry fingerprints change with it. */
#define ENGRAM_ENC_ALGO 2u     /* 2: script-split word weights, TF shaping */

#define ENGRAM_ENC_SEED 0x454E4752414D3031ull     /* "ENGRAM01" */

/* ---- TERM-FREQUENCY SHAPING, AND THE MEASUREMENT THAT ADDED IT -----------------------------------
 * BURSTINESS (Jegou, Douze & Schmid, CVPR 2009): a feature repeated many times inside one text
 * dominates that text's vector. A paragraph that says "the" ten times puts ten times the weight on it,
 * and a short query's few distinctive words become a sliver of the paragraph's norm. MEASURED on the
 * English proof corpus, held-out paragraphs: a 5% fragment (about four words) found its source only
 * 40% of the time, and the best WRONG paragraph out-scored the right one on average.
 *
 * The established remedy is signed power normalisation (Perronnin, Sanchez & Mensink, ECCV 2010):
 * each accumulated dimension x becomes sign(x)|x|^a before the L2 normalisation. Measured, same split:
 * 5% fragments 0.40 -> 0.57, 10% fragments with two typos 0.50 -> 0.69, whole paragraphs unchanged.
 *
 * ONLY EXPONENTS BUILT FROM sqrt ARE OFFERED. pow() is not required to be correctly rounded and
 * differs between C libraries in the last bit; sqrt is correctly rounded by IEEE-754. So a = 1/2 is
 * one sqrt and a = 1/4 is two, and both are bit-identical on every conforming machine (rule R5).
 *
 * MEAN-CENTRING WAS TRIED AND REJECTED. Subtracting a background mean (Mu & Viswanath, "All-but-the-
 * Top", ICLR 2018) cut the unrelated-pair cosine from 0.23 to 0.04 -- and made retrieval WORSE, 5%
 * fragments 0.40 -> 0.28. A better-looking isotropy metric bought a worse result on the actual task,
 * and it would also have made the encoder depend on corpus statistics instead of on its input alone. */
typedef enum {
    ENGRAM_TF_LINEAR  = 0,     /* x                */
    ENGRAM_TF_SQRT    = 1,     /* sign(x)|x|^(1/2) */
    ENGRAM_TF_QUARTER = 2,     /* sign(x)|x|^(1/4) */
    ENGRAM_TF_SIGN    = 3      /* sign(x): presence only */
} engram_tf;

/* A "word" is a maximal run of WORD/MARK codepoints, or ONE ideograph. A bigram of two ideographic
 * words takes w_ideo2; any other bigram -- two alphabetic words, or the seam between an alphabetic
 * word and an ideograph -- takes w_word2. */
typedef struct {
    float              w_char3;         /* default 1/16 -- a floor, see above */
    float              w_char4;         /* default 1  */
    float              w_word1;         /* default 0  -- alphabetic word unigrams */
    float              w_word2;         /* default 0  -- alphabetic word bigrams  */
    float              w_ideo1;         /* default 16 -- ideograph unigrams       */
    float              w_ideo2;         /* default 16 -- ideograph bigrams        */
    int                ordered_pairs;   /* 0: word bigrams unordered (default)            */
    engram_fold        fold;            /* ENGRAM_FOLD_COMPAT by default                   */
    engram_utf8_policy utf8;            /* ENGRAM_UTF8_REPLACE by default -- and counted   */
    uint64_t           seed;            /* ENGRAM_ENC_SEED; part of the geometry           */
    engram_tf          tf;              /* term-frequency shaping; see above               */
} engram_enc_cfg;

void      engram_enc_cfg_default(engram_enc_cfg *cfg);

/* ENGRAM_OK if every weight is finite and non-negative, at least one is positive, and the enums are
 * in range. A NaN weight would put NaN in every dimension it touched and nothing would crash. (A
 * configuration whose only positive weights are ideographic is valid: it encodes ideographic text
 * and returns E_SHORT for text with none.) */
engram_rc engram_enc_cfg_check(const engram_enc_cfg *cfg);

typedef struct {
    size_t bytes;             /* input length                                        */
    size_t codepoints;        /* decoded, before folding                             */
    size_t invalid_utf8;      /* ill-formed subparts replaced (REPLACE policy only)  */
    size_t dropped;           /* IGNORE and folded-away diacritics                   */
    size_t words;
    size_t ideographs;        /* words that were a single ideograph                   */
    size_t f_char3, f_char4, f_word1, f_word2;   /* features actually hashed (weight > 0) */
    size_t f_ideo1, f_ideo2;
    double norm;              /* L2 norm BEFORE normalisation                        */
} engram_enc_stats;

/* Encode `text` (n bytes, UTF-8) into a unit vector of dimension d.
 *   ENGRAM_OK         out holds a unit vector
 *   ENGRAM_E_SHORT    the text produced no features, or they cancelled exactly: out is all zeros,
 *                     which is NOT a unit vector and must not be stored as one
 *   ENGRAM_E_UTF8     STRICT policy and the text is ill-formed: out is all zeros
 *   ENGRAM_E_ARG      bad argument, or d not a multiple of 8 in [64, ENGRAM_ENC_DMAX]
 * cfg may be NULL (defaults); st may be NULL. Never allocates. */
engram_rc engram_encode_d(const engram_enc_cfg *cfg, const void *text, size_t n,
                          float *out, unsigned d, engram_enc_stats *st);

engram_rc engram_encode(const engram_enc_cfg *cfg, const void *text, size_t n,
                        float out[ENGRAM_D], engram_enc_stats *st);

/* THE GEOMETRY FINGERPRINT. A 64-bit hash of everything that decides where a text lands: algorithm
 * version, dimension, seed, all six weights, pair ordering, fold mode, TF shaping, the signature's
 * size and hash count, and the Unicode version. Every file that stores vectors or signatures records
 * it and refuses to be read by an encoder with a different one (rule R7):
 * a query and a stored vector built by two different functions can still produce a cosine, and it
 * will be about a different question. */
uint64_t engram_enc_geometry(const engram_enc_cfg *cfg, unsigned d);

/* ---- THE EXACT STAGE: collision-free similarity for short queries -------------------------------
 * WHY IT EXISTS. A dense vector of D dimensions is a HASHED sketch: a paragraph's hundreds of
 * features share 512 dimensions, so an unrelated paragraph overlaps a short query by chance, and
 * that noise shrinks only like 1/D. MEASURED on held-out English fragments (test_enc T8): recall@1 0.807 at D = 512,
 * 0.886 at 1024, 0.923 at 2048, 0.936 at 4096 -- and, in a development run with the earlier
 * weights, still climbing at 65536. The loss is collisions and nothing else. Storing 65536 floats per
 * memory is 256 KB; a laptop holding 100,000 memories cannot pay that.
 *
 * THE MATHEMATICS THAT REMOVES IT. With square-root shaping, a vector with NO collisions has one
 * coordinate sqrt(c_k) per distinct feature k (c_k its weighted count), so its squared norm is
 * sum_k c_k: the text's total feature weight, which a stream can add up without remembering a single
 * feature. The exact cosine of a query q and a document d is therefore
 *
 *                       sum_k sqrt(q_k * d_k)
 *      exact(q, d) = ---------------------------         the Bhattacharyya coefficient of the two
 *                     sqrt(sum_k q_k * sum_k d_k)         feature distributions (the Hellinger kernel)
 *
 * and only the QUERY's features need a table: the document streams past, each feature is looked up,
 * and its weight joins d_k if the query has k and the document total either way. Memory is bounded by
 * the query, not the document; nothing is allocated; a four-megabyte document costs one pass. It is
 * the quantity the dense vector approximates -- with no collisions the two are equal -- computed
 * without the approximation. Other TF shapings have no such closed form (their norms need every
 * c_k), which is one reason ENGRAM_TF_SQRT is the default; engram_encq_build refuses the others.
 *
 * THE CASCADE. Something cheap finds candidates among everything; the exact score orders the few
 * that matter. The first design used the dense vector to find them, and test_enc's T8 rejected it
 * (ledger W-P1.2-10); the signature below is what finds them now. */
#define ENGRAM_ENCQ_SLOTS 4096u                  /* power of two                                   */
#define ENGRAM_ENCQ_MAXF  (ENGRAM_ENCQ_SLOTS / 2u) /* distinct query features; keeps probes short  */

typedef struct {
    engram_enc_cfg cfg;                         /* the generator the query was built with (R7)    */
    unsigned       n;                           /* distinct features                              */
    double         qmass;                       /* sum of the query's feature weights             */
    uint64_t       key[ENGRAM_ENCQ_SLOTS];      /* 0 = empty                                      */
    double         qw[ENGRAM_ENCQ_SLOTS];
    double         dw[ENGRAM_ENCQ_SLOTS];       /* scratch: the current document's weights        */
    uint16_t       occ[ENGRAM_ENCQ_MAXF];       /* occupied slots, in insertion order             */
    uint16_t       sbit[ENGRAM_ENCQ_MAXF][2];   /* each feature's signature bits (ENGRAM_SIG_K)   */
    int            ready;
} engram_encq;                                  /* about 112 KB: allocate it, do not put it on a
                                                   worker thread's stack                          */

/* Build the query table.
 *   ENGRAM_OK       ready to score
 *   ENGRAM_E_SHORT  the text has no features
 *   ENGRAM_E_FULL   more than ENGRAM_ENCQ_MAXF distinct features (a query of roughly 1,000 characters
 *                   or more). The table is NOT ready. Queries that long are where the dense vector
 *                   is already near-perfect (test_enc: 15% fragments, dense recall 0.99 to 1.00),
 *                   so the caller ranks by the dense score alone -- a measured regime, not a silent
 *                   fallback.
 *   ENGRAM_E_UTF8   STRICT policy and ill-formed text
 *   ENGRAM_E_ARG    NULL q or text with n > 0, bad cfg, or cfg->tf != ENGRAM_TF_SQRT
 * cfg may be NULL (defaults). On any failure q is left not ready and every score call refuses. */
engram_rc engram_encq_build(engram_encq *q, const engram_enc_cfg *cfg, const void *text, size_t n);

/* Score one document against a ready query: *score = exact(q, d) in [0, 1].
 *   ENGRAM_OK       *score is set (0 when nothing is shared)
 *   ENGRAM_E_SHORT  the document has no features: *score = 0
 *   ENGRAM_E_UTF8   STRICT policy and the document is ill-formed: *score = 0
 *   ENGRAM_E_STATE  q is not ready
 *   ENGRAM_E_ARG    NULL q, score, or text with n > 0
 * Uses q's scratch, so one q must not be scored from two threads at once. Deterministic to the bit:
 * integer weights sum exactly, and the final sum runs over the query's features in insertion order. */
engram_rc engram_encq_score(engram_encq *q, const void *text, size_t n, double *score);

/* ---- THE SIGNATURE: which memories could contain this query? ------------------------------------
 * WHY IT EXISTS. The first cascade put the dense vector in front of the exact stage, and test_enc's
 * T8 caught it losing: on held-out English fragments the dense top 50 missed the source often enough
 * that the cascade scored 0.941 where an exhaustive exact search scored 0.952. A cosine asks "how
 * similar are these two texts overall" -- for a four-word query against a paragraph that is the
 * wrong question. The right one is CONTAINMENT: how much of the query does this memory contain?
 *
 * A signature answers it. Each memory's features are hashed into ENGRAM_SIG_BITS bits, K bits per
 * feature -- a Bloom filter (Bloom, CACM 1970), used for text as a SIGNATURE FILE (Faloutsos &
 * Christodoulakis, ACM TOIS 1984) and, bit-sliced, as the candidate generator of a web search
 * engine (BitFunnel: Goodwin et al., SIGIR 2017). A query's features are looked up in it; the
 * weight of those found, over the query's total, is the containment estimate.
 *
 *   NO FALSE NEGATIVES. A feature a memory has always finds all K of its bits set, so a memory is
 *   never scored as lacking a feature it has; a text scores exactly 1 against its own signature.
 *   FALSE POSITIVES are the price: with fill f, a feature the memory lacks still "matches" with
 *   probability f^K. A 400-character paragraph fills about 15% of 8192 bits, so f^2 = 2.3%.
 *
 * MEASURED, held-out English fragments, top 50 re-scored exactly (test_enc T8): the source reached
 * the exact stage for 99.5% of queries by signature against 97.1% by dense cosine, and the cascade
 * scored 0.955 against the dense-first cascade's 0.941 and an exhaustive exact search's 0.952. At
 * 1 KB a signature is half the size of the dense vector.
 *
 * WHAT IT IS NOT FOR. A memory of many thousands of distinct features saturates it -- every bit set,
 * every query "contained" -- so it wastes a candidate slot on every query, though the exact stage
 * then ranks it where it belongs. Memories are to be stored in bounded chunks (P1.3).
 *
 * THE BITS. h = mix64(f ^ K_SIG) for feature id f; bit i is the i-th 13-bit field from the top of h.
 * The features are those with positive weight -- exactly the ones the exact stage scores (R7). */
#define ENGRAM_SIG_LOG2  13u
#define ENGRAM_SIG_BITS  (1u << ENGRAM_SIG_LOG2)       /* 8192 bits = 1 KB */
#define ENGRAM_SIG_WORDS (ENGRAM_SIG_BITS / 64u)
#define ENGRAM_SIG_K     2u

/* Write the signature of `text` into sig (all ENGRAM_SIG_WORDS words are written).
 *   ENGRAM_OK       sig holds at least one bit
 *   ENGRAM_E_SHORT  no features: sig is all zeros (it would "contain" nothing -- store no such thing)
 *   ENGRAM_E_UTF8   STRICT policy and ill-formed text: sig is all zeros
 *   ENGRAM_E_ARG    NULL sig, NULL text with n > 0, or a bad cfg
 * cfg may be NULL (defaults); st may be NULL. Never allocates. */
engram_rc engram_encode_sig(const engram_enc_cfg *cfg, const void *text, size_t n,
                            uint64_t sig[ENGRAM_SIG_WORDS], engram_enc_stats *st);

/* The containment estimate of a ready query against one signature, in [0, 1]:
 * (sum of the weights of the query's features found) / (the query's total weight).
 *   ENGRAM_OK / ENGRAM_E_STATE (q not ready) / ENGRAM_E_ARG (NULL argument).
 * Reads q and sig only -- unlike engram_encq_score it may run on one q from many threads. */
engram_rc engram_encq_sig_score(const engram_encq *q, const uint64_t sig[ENGRAM_SIG_WORDS],
                                double *score);

/* ---- VECTOR ARITHMETIC ------------------------------------------------------------------------
 * Accumulated in double, in index order, so every result is reproducible to the bit. */
float     engram_vec_dot(const float *a, const float *b, unsigned d);
double    engram_vec_norm(const float *a, unsigned d);
engram_rc engram_vec_normalize(float *a, unsigned d);   /* ENGRAM_E_SHORT for the zero vector */

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_ENC_H */
