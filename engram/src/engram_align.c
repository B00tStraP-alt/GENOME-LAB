/* ==================================================================================================
 * engram_align.c -- the alignment. Contract in engram_align.h.
 * ============================================================================================== */
#include "engram_align.h"

engram_rc engram_align_norm(const void *text, size_t n, engram_fold fold, engram_utf8_policy policy,
                            uint32_t *out, size_t cap, size_t *len)
{
    engram_textit it;
    engram_rc rc;
    uint32_t cp;
    unsigned cls;
    size_t k = 0;
    int space = 0;
    if (len) *len = 0;
    if (!len || (!out && cap) || (!text && n)) return ENGRAM_E_ARG;
    rc = engram_textit_init(&it, text, n, fold, policy);
    if (rc != ENGRAM_OK) return rc;
    while (engram_textit_next(&it, &cp, &cls)) {
        if (cls == ENGRAM_CLASS_SPACE) { space = k > 0u; continue; }   /* leading space never counts */
        if (space) {
            if (k == cap) { *len = k; return ENGRAM_E_FULL; }
            out[k++] = 0x20u;
            space = 0;
        }
        if (k == cap) { *len = k; return ENGRAM_E_FULL; }
        out[k++] = cp;
    }
    *len = k;
    return it.err;
}

/* Rows are cue positions i = 0..qn, columns text positions j = 0..dn.
 *   D[0][j] = 0                  the stretch may start anywhere
 *   D[i][0] = i
 *   D[i][j] = min( D[i-1][j-1] + (q[i-1] != d[j-1]),     substitute (or match)
 *                  D[i-1][j] + 1,                        delete q[i-1]
 *                  D[i][j-1] + 1,                        insert d[j-1]
 *                  D[i-2][j-2] + 1   if q[i-1] == d[j-2] && q[i-2] == d[j-1] )   swap
 *   answer = min_j D[qn][j]      the stretch may end anywhere */
uint32_t engram_align_dist(const uint32_t *q, size_t qn, const uint32_t *d, size_t dn, uint32_t *work)
{
    uint32_t *pp = work, *p = work + (dn + 1u), *c = work + 2u * (dn + 1u), *t, best;
    size_t i, j;
    for (j = 0; j <= dn; j++) { pp[j] = 0u; p[j] = 0u; }
    for (i = 1; i <= qn; i++) {
        uint32_t qi = q[i - 1u];
        c[0] = (uint32_t)i;
        for (j = 1; j <= dn; j++) {
            uint32_t v = p[j - 1u] + (qi != d[j - 1u] ? 1u : 0u);
            if (p[j] + 1u < v) v = p[j] + 1u;
            if (c[j - 1u] + 1u < v) v = c[j - 1u] + 1u;
            if (i > 1u && j > 1u && qi == d[j - 2u] && q[i - 2u] == d[j - 1u] && pp[j - 2u] + 1u < v)
                v = pp[j - 2u] + 1u;
            c[j] = v;
        }
        t = pp; pp = p; p = c; c = t;
    }
    best = (uint32_t)qn;
    for (j = 0; j <= dn; j++) if (p[j] < best) best = p[j];
    return best;
}
