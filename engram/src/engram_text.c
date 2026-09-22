/* ==================================================================================================
 * engram_text.c -- strict UTF-8, O(1) classification, table-driven folding, the normalised stream.
 * ============================================================================================== */
#include "engram_text.h"
#include "engram_alloc.h"
#include "engram_buf.h"
#include "engram_unicode_tables.h"

#include <string.h>

ENGRAM_STATIC_ASSERT(ENGRAM_FOLD_MAX == ENGRAM_FOLD_OUT_MAX, fold_width_matches_tables);
ENGRAM_STATIC_ASSERT(ENGRAM_CC_WORD == ENGRAM_CLASS_WORD && ENGRAM_CC_IDEO == ENGRAM_CLASS_IDEO &&
                     ENGRAM_CC_DIACRITIC == ENGRAM_CLASS_DIACRITIC, class_ids_match_tables);

const char *engram_unicode_version(void) { return ENGRAM_UNICODE_VERSION; }

/* ==================================================================================================
 * UTF-8
 * ==================================================================================================
 * RFC 3629 well-formedness with the WHATWG maximal-subpart replacement rule. The per-lead-byte bounds
 * on the FIRST continuation byte are what exclude, without any later check:
 *
 *     E0 followed by 80..9F    overlong 3-byte forms       -> first continuation must be A0..BF
 *     ED followed by A0..BF    UTF-16 surrogates           -> first continuation must be 80..9F
 *     F0 followed by 80..8F    overlong 4-byte forms       -> first continuation must be 90..BF
 *     F4 followed by 90..BF    above U+10FFFF              -> first continuation must be 80..8F
 *     C0, C1, F5..FF           can never begin a well-formed sequence
 *
 * A continuation byte outside its bounds is NOT consumed: it ends the ill-formed subpart and is read
 * again as the start of the next sequence. That is what makes "maximal subpart" produce the same
 * number of replacement characters as every browser. */
size_t engram_utf8_decode(const uint8_t *p, size_t n, uint32_t *cp, int *valid)
{
    uint8_t b, lo = 0x80u, hi = 0xBFu;
    uint32_t c;
    size_t need, i;

    if (!p || !n) { *cp = ENGRAM_REPLACEMENT_CHAR; *valid = 0; return n ? 1u : 0u; }
    b = p[0];
    if (b < 0x80u) { *cp = b; *valid = 1; return 1u; }
    if (b >= 0xC2u && b <= 0xDFu)      { need = 1u; c = (uint32_t)(b & 0x1Fu); }
    else if (b >= 0xE0u && b <= 0xEFu) { need = 2u; c = (uint32_t)(b & 0x0Fu);
                                         if (b == 0xE0u) lo = 0xA0u;
                                         if (b == 0xEDu) hi = 0x9Fu; }
    else if (b >= 0xF0u && b <= 0xF4u) { need = 3u; c = (uint32_t)(b & 0x07u);
                                         if (b == 0xF0u) lo = 0x90u;
                                         if (b == 0xF4u) hi = 0x8Fu; }
    else { *cp = ENGRAM_REPLACEMENT_CHAR; *valid = 0; return 1u; }

    for (i = 1u; i <= need; i++) {
        if (i >= n) { *cp = ENGRAM_REPLACEMENT_CHAR; *valid = 0; return i; }   /* truncated */
        b = p[i];
        if (b < lo || b > hi) { *cp = ENGRAM_REPLACEMENT_CHAR; *valid = 0; return i; }
        lo = 0x80u;
        hi = 0xBFu;
        c = (c << 6) | (uint32_t)(b & 0x3Fu);
    }
    *cp = c;
    *valid = 1;
    return need + 1u;
}

engram_rc engram_utf8_validate(const uint8_t *p, size_t n, size_t *bad)
{
    size_t off = 0;
    if (bad) *bad = 0;
    if (!p && n) return ENGRAM_E_ARG;
    while (off < n) {
        uint32_t cp;
        int valid;
        size_t used = engram_utf8_decode(p + off, n - off, &cp, &valid);
        if (!valid) { if (bad) *bad = off; return ENGRAM_E_UTF8; }
        off += used;
    }
    return ENGRAM_OK;
}

size_t engram_utf8_encode(uint32_t cp, uint8_t out[4])
{
    if (cp < 0x80u) { out[0] = (uint8_t)cp; return 1u; }
    if (cp < 0x800u) {
        out[0] = (uint8_t)(0xC0u | (cp >> 6));
        out[1] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 2u;
    }
    if (cp >= 0xD800u && cp <= 0xDFFFu) return 0u;       /* surrogates have no UTF-8 form */
    if (cp < 0x10000u) {
        out[0] = (uint8_t)(0xE0u | (cp >> 12));
        out[1] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 3u;
    }
    if (cp <= 0x10FFFFu) {
        out[0] = (uint8_t)(0xF0u | (cp >> 18));
        out[1] = (uint8_t)(0x80u | ((cp >> 12) & 0x3Fu));
        out[2] = (uint8_t)(0x80u | ((cp >> 6) & 0x3Fu));
        out[3] = (uint8_t)(0x80u | (cp & 0x3Fu));
        return 4u;
    }
    return 0u;
}

/* ==================================================================================================
 * CLASSES
 * ==================================================================================================
 * The BMP is a two-level table: 256 block indices, and 54 distinct blocks of 256 four-bit classes.
 * One shift, one index, one nibble -- no search. Above the BMP the planes are classified by block,
 * because a full supplementary table would be large and almost entirely one value. */
unsigned engram_cp_class(uint32_t cp)
{
    if (cp < 0x10000u) {
        unsigned b = ENGRAM_CC_BLOCK[(size_t)ENGRAM_CC_INDEX[cp >> 8] * 128u + ((cp & 0xFFu) >> 1)];
        return (cp & 1u) ? (b >> 4) : (b & 0x0Fu);
    }
    if (cp > 0x10FFFFu) return ENGRAM_CLASS_PUNCT;
    if ((cp >= 0x20000u && cp <= 0x2FA1Fu) ||             /* CJK Extensions B-F, compatibility supp. */
        (cp >= 0x30000u && cp <= 0x3134Fu))               /* CJK Extension G */
        return ENGRAM_CLASS_IDEO;
    if (cp >= 0x1F000u && cp <= 0x1FAFFu) return ENGRAM_CLASS_PUNCT;   /* emoji and pictographs  */
    if (cp >= 0xE0000u && cp <= 0xE007Fu) return ENGRAM_CLASS_IGNORE;  /* tag characters         */
    if (cp >= 0xE0100u && cp <= 0xE01EFu) return ENGRAM_CLASS_IGNORE;  /* variation selectors    */
    return ENGRAM_CLASS_WORD;
}

/* ==================================================================================================
 * FOLDING
 * ============================================================================================== */
#define ENGRAM_FOLD_NRANGES (sizeof ENGRAM_FOLD_RANGES / sizeof ENGRAM_FOLD_RANGES[0])

size_t engram_cp_fold(uint32_t cp, engram_fold mode, uint32_t out[ENGRAM_FOLD_OUT_MAX])
{
    const uint32_t *tab;
    const uint16_t *pool;
    size_t i;

    if (mode != ENGRAM_FOLD_CASE && mode != ENGRAM_FOLD_COMPAT) { out[0] = cp; return 1u; }
    if (cp < 0x80u) { out[0] = (cp >= 'A' && cp <= 'Z') ? cp + 32u : cp; return 1u; }

    tab  = (mode == ENGRAM_FOLD_COMPAT) ? ENGRAM_FOLDTAB_COMPAT      : ENGRAM_FOLDTAB_CASE;
    pool = (mode == ENGRAM_FOLD_COMPAT) ? ENGRAM_FOLDTAB_COMPAT_POOL : ENGRAM_FOLDTAB_CASE_POOL;

    for (i = 0; i < ENGRAM_FOLD_NRANGES; i++) {
        const engram_fold_range *r = &ENGRAM_FOLD_RANGES[i];
        if (cp < r->lo) break;                         /* ranges are generated in ascending order */
        if (cp <= r->hi) {
            uint32_t w = tab[(size_t)r->base + (size_t)(cp - r->lo)];
            unsigned len = (unsigned)((w >> 16) & 7u), k;
            if (len == 1u) { out[0] = w & 0xFFFFu; return 1u; }
            if (len == 0u || len > ENGRAM_FOLD_OUT_MAX) break;   /* cannot happen; test_text checks */
            for (k = 0; k < len; k++) out[k] = pool[(size_t)(w & 0xFFFFu) + k];
            return len;
        }
    }
    out[0] = cp;
    return 1u;
}

/* ==================================================================================================
 * THE NORMALISED STREAM
 * ============================================================================================== */
engram_rc engram_textit_init(engram_textit *it, const void *text, size_t n,
                             engram_fold fold, engram_utf8_policy policy)
{
    if (!it) return ENGRAM_E_ARG;
    memset(it, 0, sizeof *it);
    it->err = ENGRAM_OK;
    if (!text && n) { it->err = ENGRAM_E_ARG; return ENGRAM_E_ARG; }
    if (fold != ENGRAM_FOLD_NONE && fold != ENGRAM_FOLD_CASE && fold != ENGRAM_FOLD_COMPAT) {
        it->err = ENGRAM_E_ARG;
        return ENGRAM_E_ARG;
    }
    if (policy != ENGRAM_UTF8_STRICT && policy != ENGRAM_UTF8_REPLACE) {
        it->err = ENGRAM_E_ARG;
        return ENGRAM_E_ARG;
    }
    it->p = (const uint8_t *)text;
    it->n = text ? n : 0u;
    it->fold = fold;
    it->policy = policy;
    return ENGRAM_OK;
}

int engram_textit_next(engram_textit *it, uint32_t *cp, unsigned *cls)
{
    if (!it || !cp || !cls) return 0;
    for (;;) {
        if (it->ipend < it->npend) {
            uint32_t c = it->pend[it->ipend++];
            unsigned k = engram_cp_class(c);
            if (k == ENGRAM_CLASS_IGNORE) { it->n_dropped++; continue; }
            if (k == ENGRAM_CLASS_DIACRITIC) {
                if (it->fold == ENGRAM_FOLD_COMPAT) { it->n_dropped++; continue; }
                k = ENGRAM_CLASS_MARK;
            }
            *cp = c;
            *cls = k;
            return 1;
        }
        if (it->err != ENGRAM_OK || it->off >= it->n) return 0;
        if (it->p[it->off] < 0x80u) {
            /* ASCII FAST PATH: the same answer the general path gives (a one-byte scalar is always
             * valid; its fold is itself or, under CASE and COMPAT, its lowercase; it never
             * expands), without three calls per byte. test_text's exhaustive vectors cover it. */
            uint32_t c = it->p[it->off++];
            unsigned k;
            it->n_codepoints++;
            if (it->fold != ENGRAM_FOLD_NONE && c >= 'A' && c <= 'Z') c += 32u;
            k = engram_cp_class(c);
            if (k == ENGRAM_CLASS_IGNORE) { it->n_dropped++; continue; }
            if (k == ENGRAM_CLASS_DIACRITIC) {
                if (it->fold == ENGRAM_FOLD_COMPAT) { it->n_dropped++; continue; }
                k = ENGRAM_CLASS_MARK;
            }
            *cp = c;
            *cls = k;
            return 1;
        }
        {
            uint32_t c;
            int valid;
            size_t used = engram_utf8_decode(it->p + it->off, it->n - it->off, &c, &valid);
            it->off += used;
            it->n_codepoints++;
            if (!valid) {
                if (it->policy == ENGRAM_UTF8_STRICT) { it->err = ENGRAM_E_UTF8; return 0; }
                it->n_invalid++;
            }
            it->npend = (unsigned)engram_cp_fold(c, it->fold, it->pend);
            it->ipend = 0u;
        }
    }
}

engram_rc engram_text_normalize(const void *text, size_t n, engram_fold fold,
                                engram_utf8_policy policy, char **out, size_t *len)
{
    engram_textit it;
    engram_wbuf w;
    uint32_t cp;
    unsigned k;
    int pending_space = 0, any = 0;
    uint8_t enc[4];
    uint8_t *bytes = NULL;
    size_t blen = 0;
    engram_rc rc;

    if (out) *out = NULL;
    if (len) *len = 0;
    if (!out || !len) return ENGRAM_E_ARG;
    rc = engram_textit_init(&it, text, n, fold, policy);
    if (rc != ENGRAM_OK) return rc;

    engram_wbuf_init(&w);
    while (engram_textit_next(&it, &cp, &k)) {
        size_t nb;
        if (k == ENGRAM_CLASS_SPACE) { if (any) pending_space = 1; continue; }
        if (pending_space) { engram_wbuf_u8(&w, (uint8_t)' '); pending_space = 0; }
        nb = engram_utf8_encode(cp, enc);
        if (!nb) { engram_wbuf_free(&w); return ENGRAM_E_INTERNAL; }   /* fold produced a surrogate */
        engram_wbuf_bytes(&w, enc, nb);
        any = 1;
    }
    if (it.err != ENGRAM_OK) { engram_wbuf_free(&w); return it.err; }
    engram_wbuf_u8(&w, 0u);
    rc = engram_wbuf_finish(&w, &bytes, &blen);
    if (rc != ENGRAM_OK) return rc;
    *out = (char *)bytes;
    *len = blen - 1u;
    return ENGRAM_OK;
}
