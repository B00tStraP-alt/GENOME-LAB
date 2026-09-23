/* ==================================================================================================
 * engram_chunk.c -- the episode cutter. Contract and guarantees in engram_chunk.h.
 * ============================================================================================== */
#include "engram_chunk.h"
#include "engram_text.h"

#include <string.h>

/* One decoder step: the codepoint at off, its byte length, and whether it is whitespace. Ill-formed
 * bytes decode as one maximal subpart (U+FFFD) and are never whitespace -- they are content the chunk
 * must keep, whole. */
static size_t engram_chunk_step(const uint8_t *p, size_t n, size_t off, uint32_t *cp)
{
    int valid;
    size_t used = engram_utf8_decode(p + off, n - off, cp, &valid);
    return used ? used : 1u;
}

static int engram_is_space(uint32_t cp) { return engram_cp_class(cp) == ENGRAM_CLASS_SPACE; }

/* Sentence enders. The CJK and fullwidth ones end a sentence whatever follows, because those scripts
 * are written without spaces; the ASCII ones only when a space, a closer, or the end follows, so the
 * full stop in "3.14" or "e.g." mid-sentence is not taken for one. */
static int engram_is_cjk_end(uint32_t cp)
{
    return cp == 0x3002u || cp == 0xFF01u || cp == 0xFF1Fu || cp == 0xFF1Bu || cp == 0xFF0Eu ||
           cp == 0x2026u;
}

static int engram_is_ascii_end(uint32_t cp) { return cp == '.' || cp == '!' || cp == '?' || cp == ';'; }

static int engram_is_closer(uint32_t cp)
{
    return cp == '"' || cp == '\'' || cp == ')' || cp == ']' || cp == 0x00BBu || cp == 0x201Du ||
           cp == 0x2019u || cp == 0x300Du || cp == 0x300Fu || cp == 0xFF09u || cp == 0x3011u;
}

engram_rc engram_chunkit_init(engram_chunkit *it, const void *text, size_t n, size_t cap, size_t overlap)
{
    if (!it) return ENGRAM_E_ARG;
    memset(it, 0, sizeof *it);
    if (!text && n) return ENGRAM_E_ARG;
    if (cap < ENGRAM_CHUNK_MIN_CAP || cap > ENGRAM_EPI_TAIL || overlap >= cap / 2u) return ENGRAM_E_ARG;
    it->p = (const uint8_t *)text;
    it->n = text ? n : 0u;
    it->cap = cap;
    it->overlap = overlap;
    return ENGRAM_OK;
}

/* Skip whitespace forward from off; returns the first non-space decoder step at or after it. */
static size_t engram_skip_space(const uint8_t *p, size_t n, size_t off)
{
    while (off < n) {
        uint32_t cp;
        size_t used = engram_chunk_step(p, n, off, &cp);
        if (!engram_is_space(cp)) break;
        off += used;
    }
    return off;
}

int engram_chunkit_next(engram_chunkit *it, size_t *off, size_t *len)
{
    /* Decoder-step boundaries inside the window, relative to start, and whether the codepoint BEFORE
     * each boundary was whitespace (so the boundary begins a word). cap <= ENGRAM_EPI_TAIL, and every
     * step is at least one byte, so cap + 1 entries always suffice. */
    uint16_t bnd[ENGRAM_EPI_TAIL + 1u];
    uint8_t  after_space[ENGRAM_EPI_TAIL + 1u];
    const uint8_t *p;
    size_t n, start, pos, lim, minfill, end, cut_para = 0, cut_sent = 0, cut_space = 0, cut_hard = 0, nb = 0;
    uint32_t cp, prev_cp = 0;
    int kind;

    if (!it || !off || !len || !it->p) return 0;
    p = it->p;
    n = it->n;
    start = engram_skip_space(p, n, it->pos);
    if (start >= n) { it->pos = n; return 0; }
    if (it->started && start <= it->last_start) start = engram_skip_space(p, n, it->last_start + 1u);
    if (start >= n) { it->pos = n; return 0; }

    lim = start + it->cap;
    minfill = it->cap / 4u;
    pos = start;
    bnd[nb] = 0; after_space[nb] = 0; nb++;
    while (pos < n) {
        size_t used = engram_chunk_step(p, n, pos, &cp);
        size_t b = pos + used;
        if (b > lim) break;
        bnd[nb] = (uint16_t)(b - start);
        after_space[nb] = (uint8_t)engram_is_space(cp);
        nb++;
        if (b - start >= minfill) {
            uint32_t nx = 0;
            int at_end = (b >= n);
            if (!at_end) (void)engram_chunk_step(p, n, b, &nx);
            if (cp == '\n' && !at_end) {                     /* a paragraph break: \n, spaces, \n */
                size_t q = b;
                while (q < n && q < lim) {                   /* bounded: a hostile whitespace run
                                                                beyond the window costs nothing */
                    uint32_t c2;
                    size_t u2 = engram_chunk_step(p, n, q, &c2);
                    if (c2 == '\n') { cut_para = pos; break; }
                    if (!engram_is_space(c2)) break;
                    q += u2;
                }
            }
            if (engram_is_cjk_end(cp) ||
                (engram_is_ascii_end(cp) && (at_end || engram_is_space(nx) || engram_is_closer(nx))))
                cut_sent = b;
            if (engram_is_closer(cp) && (engram_is_cjk_end(prev_cp) || engram_is_ascii_end(prev_cp)) &&
                cut_sent == pos)
                cut_sent = b;                                /* keep the closing quote with its sentence */
            if (engram_is_space(cp) && !engram_is_space(prev_cp)) cut_space = pos;
        }
        cut_hard = b;
        prev_cp = cp;
        pos = b;
    }

    if (pos >= n && n - start <= it->cap) { end = n; kind = 0; }
    else if (cut_para > start) { end = cut_para; kind = 1; }
    else if (cut_sent > start) { end = cut_sent; kind = 2; }
    else if (cut_space > start) { end = cut_space; kind = 3; }
    else { end = cut_hard; kind = 4; }
    if (end <= start) end = cut_hard > start ? cut_hard : start + 1u;   /* cannot happen: cap >= 32 */

    /* trim trailing whitespace: walk the recorded boundaries back while the step before is a space */
    {
        size_t k = nb;
        while (k > 1u && start + bnd[k - 1u] > end) k--;
        while (k > 1u && start + bnd[k - 1u] == end && after_space[k - 1u]) {
            k--;
            end = start + bnd[k - 1u];
        }
    }

    *off = start;
    *len = end - start;
    it->n_chunks++;
    if (kind == 1) it->n_paragraph++;
    else if (kind == 2) it->n_sentence++;
    else if (kind == 3) it->n_space++;
    else if (kind == 4) it->n_hard++;
    it->last_start = start;
    it->started = 1;

    /* where the next chunk may begin */
    if (kind == 0) { it->pos = n; return 1; }
    it->pos = end;
    if (it->overlap) {
        size_t want = end > start + it->overlap ? end - it->overlap : start + 1u, k;
        for (k = 1; k < nb; k++) {
            size_t b = start + bnd[k];
            if (b >= end) break;
            if (b >= want && after_space[k]) { it->pos = b; break; }   /* the first word start at or after want */
        }
    }
    return 1;
}
