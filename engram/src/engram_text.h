/* ==================================================================================================
 * engram_text.h -- UTF-8, character classes, folding, and the normalised stream the encoder reads.
 * ==================================================================================================
 *
 * WHAT A PERSON'S MEMORY IS WRITTEN IN
 * ==================================================================================================
 * Whatever they type. That is French with accents, German with sharp s, Vietnamese with stacked
 * diacritics, Russian, Greek, Chinese without spaces, Hindi with vowel signs, and English with
 * apostrophes -- often in the same notes. An encoder that lowercases ASCII bytes and treats everything
 * else as opaque serves one of those people. This layer serves all of them, within stated limits.
 *
 * ==================================================================================================
 * THE NORMALISED STREAM
 * ==================================================================================================
 * engram_textit walks UTF-8 and yields (codepoint, class) pairs after folding:
 *
 *   FOLD_NONE     codepoints exactly as written
 *   FOLD_CASE     full Unicode case folding in the covered scripts (so "STRASSE" == "strasse" and
 *                 sharp-s folds to "ss")
 *   FOLD_COMPAT   case folding PLUS compatibility decomposition with diacritics removed: "Ecole" ==
 *                 "ecole" with an acute, "Viet Nam" matches its fully-accented spelling, a fullwidth
 *                 "ABC" matches "abc", ligature "fi" matches "fi", a circled one matches "1".
 *
 * IGNORE-class characters are never yielded (apostrophes, zero-width joiners, variation selectors).
 * DIACRITIC-class characters are dropped under FOLD_COMPAT and yielded as MARK otherwise.
 *
 * ==================================================================================================
 * WHAT IS NOT COVERED, STATED RATHER THAN DISCOVERED
 * ==================================================================================================
 *   - Folding covers Latin, Greek, Cyrillic, Armenian, the fullwidth block, superscripts, enclosed
 *     alphanumerics and the Latin ligatures. Other cased scripts (Georgian, Cherokee, Deseret, ...)
 *     pass through unfolded.
 *   - Thai, Lao, Khmer and Myanmar are written without spaces and have no dictionary segmentation
 *     here: a run of Thai letters is one "word". Character n-grams still match inside it.
 *   - Supplementary-plane characters are classified by block (CJK extensions are ideographs, emoji
 *     are punctuation, tag characters are ignored) and are never folded.
 *
 * ==================================================================================================
 * INVALID UTF-8
 * ==================================================================================================
 * STRICT refuses the text (ENGRAM_E_UTF8). REPLACE substitutes U+FFFD for each MAXIMAL SUBPART of an
 * ill-formed sequence -- the WHATWG Encoding Standard's rule, which every browser implements -- and
 * COUNTS every substitution, so a replaced byte is never a silent one. Overlong forms, surrogates
 * (U+D800..U+DFFF) and anything above U+10FFFF are ill-formed, per RFC 3629.
 * ============================================================================================== */
#ifndef ENGRAM_TEXT_H
#define ENGRAM_TEXT_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- UTF-8 ----------------------------------------------------------------------------------- */
#define ENGRAM_REPLACEMENT_CHAR 0xFFFDu

/* Decode one codepoint from p[0..n), n > 0. Returns the number of bytes consumed (always >= 1).
 * *valid is 1 for a well-formed sequence (and *cp is its codepoint), or 0 for an ill-formed maximal
 * subpart (and *cp is U+FFFD). Never reads past n. */
size_t engram_utf8_decode(const uint8_t *p, size_t n, uint32_t *cp, int *valid);

/* ENGRAM_OK if all n bytes are well-formed UTF-8; else ENGRAM_E_UTF8 with *bad set to the offset of
 * the first ill-formed byte (bad may be NULL). */
engram_rc engram_utf8_validate(const uint8_t *p, size_t n, size_t *bad);

/* Encode cp into out, returning its length 1..4, or 0 if cp is a surrogate or above U+10FFFF. */
size_t engram_utf8_encode(uint32_t cp, uint8_t out[4]);

/* ---- CLASSES ---------------------------------------------------------------------------------- */
#define ENGRAM_CLASS_WORD      0u
#define ENGRAM_CLASS_SPACE     1u
#define ENGRAM_CLASS_PUNCT     2u
#define ENGRAM_CLASS_MARK      3u
#define ENGRAM_CLASS_IDEO      4u
#define ENGRAM_CLASS_IGNORE    5u
#define ENGRAM_CLASS_DIACRITIC 6u

unsigned engram_cp_class(uint32_t cp);

/* The Unicode Character Database edition the tables were generated from. */
const char *engram_unicode_version(void);

/* ---- FOLDING ---------------------------------------------------------------------------------- */
typedef enum {
    ENGRAM_FOLD_NONE   = 0,
    ENGRAM_FOLD_CASE   = 1,
    ENGRAM_FOLD_COMPAT = 2
} engram_fold;

#define ENGRAM_FOLD_OUT_MAX 4u    /* the longest fold in the tables: a parenthesised ten, "(10)" */

/* Fold one codepoint. Writes 1..ENGRAM_FOLD_OUT_MAX codepoints to out and returns how many. A mode
 * outside the enum is treated as FOLD_NONE. */
size_t engram_cp_fold(uint32_t cp, engram_fold mode, uint32_t out[ENGRAM_FOLD_OUT_MAX]);

/* ---- THE NORMALISED STREAM ------------------------------------------------------------------- */
typedef enum {
    ENGRAM_UTF8_STRICT  = 0,
    ENGRAM_UTF8_REPLACE = 1
} engram_utf8_policy;

typedef struct {
    const uint8_t      *p;
    size_t              n, off;
    engram_fold         fold;
    engram_utf8_policy  policy;
    uint32_t            pend[ENGRAM_FOLD_OUT_MAX];
    unsigned            npend, ipend;
    size_t              n_codepoints;   /* decoded, before folding                    */
    size_t              n_invalid;      /* ill-formed subparts replaced under REPLACE  */
    size_t              n_dropped;      /* IGNORE and folded-away DIACRITIC characters */
    engram_rc           err;
} engram_textit;

engram_rc engram_textit_init(engram_textit *it, const void *text, size_t n,
                             engram_fold fold, engram_utf8_policy policy);

/* Yield the next normalised codepoint and its class (never IGNORE, never DIACRITIC). Returns 1 on a
 * codepoint, 0 at the end OR on error -- check it->err, which is ENGRAM_E_UTF8 if STRICT met an
 * ill-formed sequence. */
int engram_textit_next(engram_textit *it, uint32_t *cp, unsigned *cls);

/* The whole normalised text as UTF-8: folded, IGNORE characters removed, every run of SPACE collapsed
 * to one ASCII space, leading and trailing space trimmed. *out is engram_alloc'd (free with
 * engram_free) and NUL-terminated; *len excludes the terminator. For display, debugging and tests: it
 * is exactly the character stream the encoder hashes, minus the boundary spaces it adds. */
engram_rc engram_text_normalize(const void *text, size_t n, engram_fold fold,
                                engram_utf8_policy policy, char **out, size_t *len);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_TEXT_H */
