/* ==================================================================================================
 * engram_cascade.c -- the Cascade weight. Contract in engram_cascade.h.
 * ============================================================================================== */
#include "engram_cascade.h"
#include "engram_alloc.h"
#include "engram_buf.h"
#include "engram_math.h"

#include <math.h>
#include <string.h>

#define ENGRAM_CASCADE_INIT_STEP 0xFFFFFFFFull

float engram_cascade_uniform(uint64_t seed, uint64_t step, uint64_t index)
{
    uint64_t base = engram_mix64(seed * 0x9E3779B97F4A7C15ull + step);
    return (float)(engram_mix64(index ^ base) >> 40) * (1.0f / 16777216.0f);
}

engram_rc engram_cascade_init(engram_cascade *m, uint32_t rows, uint32_t cols, uint32_t K, float kappa,
                              uint64_t seed)
{
    size_t n, i;
    uint32_t d;
    if (!m) return ENGRAM_E_ARG;
    memset(m, 0, sizeof *m);
    if (!rows || !cols || (K != 4u && K != 16u && K != 64u) || !(kappa >= 0.0f && kappa <= 16.0f))
        return ENGRAM_E_ARG;
    if ((size_t)rows > (size_t)-1 / cols) return ENGRAM_E_OVERFLOW;
    n = (size_t)rows * cols;
    m->pos = (uint8_t *)engram_array(n, 1u);
    m->val = (int8_t *)engram_array(n, 1u);
    if (!m->pos || !m->val) { engram_cascade_free(m); return ENGRAM_E_MEM; }
    m->rows = rows; m->cols = cols; m->K = K; m->kappa = kappa; m->seed = seed; m->step = 0;
    for (i = 0; i < n; i++) {
        /* the prototype: min(floor(u * 3K), 3K - 1) with u a 24-bit float -- exact in float */
        float u = engram_cascade_uniform(seed, ENGRAM_CASCADE_INIT_STEP, i) * (float)(3u * K);
        uint32_t p = (uint32_t)u;
        if (p > 3u * K - 1u) p = 3u * K - 1u;
        m->pos[i] = (uint8_t)p;
        m->val[i] = (int8_t)engram_cascade_value(p, K);
    }
    /* 2^(-kappa * d * 4 / K), from + - * / alone (engram_math.h) */
    for (d = 0; d < K; d++)
        m->meta[d] = (float)engram_exp2(-(double)kappa * (double)d * 4.0 / (double)K);
    return ENGRAM_OK;
}

void engram_cascade_free(engram_cascade *m)
{
    if (!m) return;
    engram_free(m->pos);
    engram_free(m->val);
    memset(m, 0, sizeof *m);
}

/* One weight's move: the body both updates share. u is the desired move in positions. */
static void engram_cascade_move(engram_cascade *m, size_t cell, float u, uint64_t step, uint64_t key,
                                uint64_t *mv, uint64_t *fl)
{
    const uint32_t K = m->K, top = 3u * m->K - 1u;
    float mag = u < 0.0f ? -u : u, whole;
    uint32_t p = m->pos[cell], steps;
    int up = u > 0.0f, v = engram_cascade_value(p, K);
    if (mag > (float)K) mag = (float)K;
    if (mag == 0.0f) return;
    if ((v > 0 && !up) || (v < 0 && up)) mag *= m->meta[engram_cascade_depth(p, K)];
    whole = floorf(mag);
    steps = (uint32_t)whole + (engram_cascade_uniform(m->seed, step, key) < mag - whole ? 1u : 0u);
    if (!steps) return;
    if (up) p = p + steps > top ? top : p + steps;
    else p = p < steps ? 0u : p - steps;
    if (p != m->pos[cell]) {
        int nv = engram_cascade_value(p, K);
        (*mv)++;
        if (nv != v) (*fl)++;
        m->pos[cell] = (uint8_t)p;
        m->val[cell] = (int8_t)nv;
    }
}

void engram_cascade_update_rows(engram_cascade *m, const float *g, float lr, uint64_t step, uint32_t r0,
                                uint32_t r1, uint64_t *moved, uint64_t *flipped)
{
    const uint32_t cols = m->cols;
    uint64_t mv = 0, fl = 0;
    uint32_t r, c;
    for (r = r0; r < r1 && r < m->rows; r++) {
        const float *gr = g + (size_t)r * cols;
        double ss = 0.0;
        float inv;
        for (c = 0; c < cols; c++) ss += (double)gr[c] * (double)gr[c];
        /* rms + 1e-12, as the prototype; an all-zero row moves nothing */
        inv = (float)(1.0 / (sqrt(ss / (double)cols) + 1e-12));
        for (c = 0; c < cols; c++)
            engram_cascade_move(m, (size_t)r * cols + c, (-lr * gr[c]) * inv, step, (uint64_t)r * cols + c, &mv, &fl);
    }
    if (moved) *moved += mv;
    if (flipped) *flipped += fl;
}

void engram_cascade_update_units(engram_cascade *m, const uint32_t *feat, size_t n_feat, const float *g, float lr,
                                 uint64_t step, uint32_t h0, uint32_t h1, uint64_t *moved, uint64_t *flipped)
{
    const uint32_t H = m->cols, F = m->rows;
    uint64_t mv = 0, fl = 0;
    uint32_t h;
    size_t i;
    if (h1 > H) h1 = H;
    for (h = h0; h < h1; h++) {
        double ss = 0.0;
        float inv;
        for (i = 0; i < n_feat; i++) { double x = (double)g[i * H + h]; ss += x * x; }
        inv = (float)(1.0 / (sqrt(ss / (double)F) + 1e-12));
        for (i = 0; i < n_feat; i++) {
            if (feat[i] >= F) continue;
            engram_cascade_move(m, (size_t)feat[i] * H + h, (-lr * g[i * H + h]) * inv, step,
                                (uint64_t)h * F + feat[i], &mv, &fl);
        }
    }
    if (moved) *moved += mv;
    if (flipped) *flipped += fl;
}

void engram_cascade_commit_step(engram_cascade *m, uint64_t step)
{
    if (m && step > m->step) m->step = step;
}

uint64_t engram_cascade_fingerprint(const engram_cascade *m)
{
    uint64_t h;
    uint32_t kb;
    if (!m || !m->pos) return 0u;
    memcpy(&kb, &m->kappa, sizeof kb);
    h = engram_mix2(0x43415343ull, ((uint64_t)m->rows << 32) | m->cols);
    h = engram_mix2(h, ((uint64_t)m->K << 32) | kb);
    h = engram_mix2(h, m->seed);
    h = engram_mix2(h, m->step);
    return engram_mix2(h, engram_hash_bytes(m->pos, (size_t)m->rows * m->cols, 5u));
}

void engram_cascade_matvec(const engram_cascade *m, const int8_t *x, int32_t *y, uint32_t r0, uint32_t r1)
{
    uint32_t r, c;
    for (r = r0; r < r1 && r < m->rows; r++) {
        const int8_t *v = m->val + (size_t)r * m->cols;
        int32_t acc = 0;
        for (c = 0; c < m->cols; c++) acc += (int32_t)v[c] * (int32_t)x[c];
        y[r] = acc;
    }
}

void engram_cascade_matvec_t(const engram_cascade *m, const float *d, float *y, uint32_t c0, uint32_t c1)
{
    uint32_t r, c;
    if (c1 > m->cols) c1 = m->cols;
    for (c = c0; c < c1; c++) y[c] = 0.0f;
    for (r = 0; r < m->rows; r++) {
        const int8_t *v = m->val + (size_t)r * m->cols;
        float dr = d[r];
        if (dr == 0.0f) continue;
        for (c = c0; c < c1; c++) {
            if (v[c] > 0) y[c] += dr;
            else if (v[c] < 0) y[c] -= dr;
        }
    }
}

engram_rc engram_cascade_gather(const engram_cascade *m, const uint32_t *rows, size_t n, int32_t *y)
{
    size_t i;
    uint32_t c;
    if (!m || !m->val || (!rows && n) || !y) return ENGRAM_E_ARG;
    for (i = 0; i < n; i++) if (rows[i] >= m->rows) return ENGRAM_E_ARG;
    for (c = 0; c < m->cols; c++) y[c] = 0;
    for (i = 0; i < n; i++) {
        const int8_t *v = m->val + (size_t)rows[i] * m->cols;
        for (c = 0; c < m->cols; c++) y[c] += v[c];
    }
    return ENGRAM_OK;
}

size_t engram_cascade_size(const engram_cascade *m)
{
    return 4u * 4u + 4u + 2u * 8u + (size_t)m->rows * m->cols;
}

void engram_cascade_write(const engram_cascade *m, uint8_t *out)
{
    uint32_t kb;
    memcpy(&kb, &m->kappa, sizeof kb);
    engram_le_put_u32(out + 0, 1u);
    engram_le_put_u32(out + 4, m->rows);
    engram_le_put_u32(out + 8, m->cols);
    engram_le_put_u32(out + 12, m->K);
    engram_le_put_u32(out + 16, kb);
    engram_le_put_u64(out + 20, m->seed);
    engram_le_put_u64(out + 28, m->step);
    memcpy(out + 36, m->pos, (size_t)m->rows * m->cols);
}

engram_rc engram_cascade_read(engram_cascade *m, const uint8_t *p, size_t n, uint32_t max_rows, uint32_t max_cols,
                              size_t *used)
{
    uint32_t rows, cols, K, kb;
    float kappa;
    size_t cells, i;
    engram_rc rc;
    if (!m || !p || !used) return ENGRAM_E_ARG;
    *used = 0;
    if (n < 36u) return ENGRAM_E_FORMAT;
    if (engram_le_get_u32(p) != 1u) return ENGRAM_E_VERSION;
    rows = engram_le_get_u32(p + 4);
    cols = engram_le_get_u32(p + 8);
    K = engram_le_get_u32(p + 12);
    kb = engram_le_get_u32(p + 16);
    memcpy(&kappa, &kb, sizeof kappa);
    if (!rows || !cols || rows > max_rows || cols > max_cols || (K != 4u && K != 16u && K != 64u) ||
        !(kappa >= 0.0f && kappa <= 16.0f))
        return ENGRAM_E_FORMAT;
    cells = (size_t)rows * cols;
    if (cells > n - 36u) return ENGRAM_E_FORMAT;
    for (i = 0; i < cells; i++) if (p[36u + i] >= 3u * K) return ENGRAM_E_FORMAT;
    rc = engram_cascade_init(m, rows, cols, K, kappa, engram_le_get_u64(p + 20));
    if (rc != ENGRAM_OK) return rc == ENGRAM_E_MEM ? rc : ENGRAM_E_FORMAT;
    m->step = engram_le_get_u64(p + 28);
    memcpy(m->pos, p + 36, cells);
    for (i = 0; i < cells; i++) m->val[i] = (int8_t)engram_cascade_value(m->pos[i], K);
    *used = 36u + cells;
    return ENGRAM_OK;
}
