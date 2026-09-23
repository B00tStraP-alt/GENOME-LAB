/* ==================================================================================================
 * engram_buf.c -- little-endian, bounded, sticky serialisation. The contract is in engram_buf.h.
 * ============================================================================================== */
#include "engram_buf.h"
#include "engram_alloc.h"

#include <string.h>

/* A non-NULL pointer for zero-length views, so NULL can keep meaning "failed". */
static const uint8_t ENGRAM_BUF_EMPTY[1] = { 0 };

/* ---- RAW HELPERS -------------------------------------------------------------------------------- */
void engram_le_put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

void engram_le_put_u64(uint8_t *p, uint64_t v)
{
    engram_le_put_u32(p, (uint32_t)v);
    engram_le_put_u32(p + 4, (uint32_t)(v >> 32));
}

uint32_t engram_le_get_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint64_t engram_le_get_u64(const uint8_t *p)
{
    return (uint64_t)engram_le_get_u32(p) | ((uint64_t)engram_le_get_u32(p + 4) << 32);
}

/* The IEEE-754 test by bit pattern: all-ones exponent is NaN or infinity. Exact, needs no libm, and
 * cannot be perturbed by the floating-point environment or by -ffast-math, which is entitled to
 * assume isfinite() is always true and fold it away. */
static int engram_f32_bits_finite(uint32_t u) { return (u & 0x7F800000u) != 0x7F800000u; }

/* ==================================================================================================
 * WRITER
 * ============================================================================================== */
void engram_wbuf_init(engram_wbuf *b)
{
    if (!b) return;
    b->p = NULL;
    b->n = 0;
    b->cap = 0;
    b->err = ENGRAM_OK;
}

void engram_wbuf_free(engram_wbuf *b)
{
    if (!b) return;
    engram_free(b->p);
    b->p = NULL;
    b->n = 0;
    b->cap = 0;
    /* err is deliberately kept: freeing a failed buffer must not launder its failure. */
}

static void engram_wb_fail(engram_wbuf *b, engram_rc rc)
{
    if (b && b->err == ENGRAM_OK && rc != ENGRAM_OK) b->err = rc;
}

/* 1 if `extra` more bytes fit (growing if needed), else 0 with the error recorded. */
static int engram_wb_room(engram_wbuf *b, size_t extra)
{
    void *pp;
    size_t cap;
    engram_rc rc;
    if (!b || b->err != ENGRAM_OK) return 0;
    if (extra > SIZE_MAX - b->n) { engram_wb_fail(b, ENGRAM_E_OVERFLOW); return 0; }
    if (b->p && b->n + extra <= b->cap) return 1;
    pp = b->p;
    cap = b->cap;
    rc = engram_grow(&pp, &cap, b->n + extra, 1u);
    if (rc != ENGRAM_OK) { engram_wb_fail(b, rc); return 0; }
    b->p = (uint8_t *)pp;
    b->cap = cap;
    return 1;
}

void engram_wbuf_reserve(engram_wbuf *b, size_t extra) { (void)engram_wb_room(b, extra); }

void engram_wbuf_bytes(engram_wbuf *b, const void *src, size_t n)
{
    if (!n) return;
    if (!src) { engram_wb_fail(b, ENGRAM_E_ARG); return; }
    if (!engram_wb_room(b, n)) return;
    memcpy(b->p + b->n, src, n);
    b->n += n;
}

void engram_wbuf_zeros(engram_wbuf *b, size_t n)
{
    if (!n) return;
    if (!engram_wb_room(b, n)) return;
    memset(b->p + b->n, 0, n);
    b->n += n;
}

void engram_wbuf_u8(engram_wbuf *b, uint8_t v)
{
    if (!engram_wb_room(b, 1u)) return;
    b->p[b->n++] = v;
}

void engram_wbuf_u16(engram_wbuf *b, uint16_t v)
{
    if (!engram_wb_room(b, 2u)) return;
    b->p[b->n++] = (uint8_t)(v);
    b->p[b->n++] = (uint8_t)(v >> 8);
}

void engram_wbuf_u32(engram_wbuf *b, uint32_t v)
{
    if (!engram_wb_room(b, 4u)) return;
    engram_le_put_u32(b->p + b->n, v);
    b->n += 4u;
}

void engram_wbuf_u64(engram_wbuf *b, uint64_t v)
{
    if (!engram_wb_room(b, 8u)) return;
    engram_le_put_u64(b->p + b->n, v);
    b->n += 8u;
}

void engram_wbuf_i32(engram_wbuf *b, int32_t v) { engram_wbuf_u32(b, (uint32_t)v); }
void engram_wbuf_i64(engram_wbuf *b, int64_t v) { engram_wbuf_u64(b, (uint64_t)v); }

void engram_wbuf_f32(engram_wbuf *b, float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    engram_wbuf_u32(b, u);
}

void engram_wbuf_f64(engram_wbuf *b, double v)
{
    uint64_t u;
    memcpy(&u, &v, sizeof u);
    engram_wbuf_u64(b, u);
}

void engram_wbuf_f32s(engram_wbuf *b, const float *v, size_t n)
{
    size_t i;
    if (!n) return;
    if (!v) { engram_wb_fail(b, ENGRAM_E_ARG); return; }
    if (n > SIZE_MAX / 4u) { engram_wb_fail(b, ENGRAM_E_OVERFLOW); return; }
    if (!engram_wb_room(b, n * 4u)) return;
    for (i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &v[i], sizeof u);
        engram_le_put_u32(b->p + b->n, u);
        b->n += 4u;
    }
}

void engram_wbuf_blob(engram_wbuf *b, const void *p, size_t n)
{
    if (n > (size_t)UINT32_MAX) { engram_wb_fail(b, ENGRAM_E_OVERFLOW); return; }
    if (n && !p) { engram_wb_fail(b, ENGRAM_E_ARG); return; }
    engram_wbuf_u32(b, (uint32_t)n);
    engram_wbuf_bytes(b, p, n);
}

void engram_wbuf_str(engram_wbuf *b, const char *s)
{
    if (!s) { engram_wb_fail(b, ENGRAM_E_ARG); return; }
    engram_wbuf_blob(b, s, strlen(s));
}

void engram_wbuf_put_u32_at(engram_wbuf *b, size_t off, uint32_t v)
{
    if (!b || b->err != ENGRAM_OK) return;
    if (off > b->n || b->n - off < 4u) { engram_wb_fail(b, ENGRAM_E_INTERNAL); return; }
    engram_le_put_u32(b->p + off, v);
}

void engram_wbuf_put_u64_at(engram_wbuf *b, size_t off, uint64_t v)
{
    if (!b || b->err != ENGRAM_OK) return;
    if (off > b->n || b->n - off < 8u) { engram_wb_fail(b, ENGRAM_E_INTERNAL); return; }
    engram_le_put_u64(b->p + off, v);
}

size_t engram_wbuf_len(const engram_wbuf *b) { return b ? b->n : 0; }

engram_rc engram_wbuf_status(const engram_wbuf *b) { return b ? b->err : ENGRAM_E_ARG; }

engram_rc engram_wbuf_finish(engram_wbuf *b, uint8_t **out, size_t *len)
{
    engram_rc rc;
    if (out) *out = NULL;
    if (len) *len = 0;
    if (!b) return ENGRAM_E_ARG;
    if (!out || !len) { engram_wbuf_free(b); return ENGRAM_E_ARG; }
    rc = b->err;
    if (rc != ENGRAM_OK) { engram_wbuf_free(b); return rc; }
    if (!b->p) {
        b->p = (uint8_t *)engram_malloc(0);
        if (!b->p) { engram_wbuf_free(b); return ENGRAM_E_MEM; }
    }
    *out = b->p;
    *len = b->n;
    b->p = NULL;
    b->n = 0;
    b->cap = 0;
    return ENGRAM_OK;
}

/* ==================================================================================================
 * READER
 * ============================================================================================== */
void engram_rbuf_init(engram_rbuf *r, const void *p, size_t n)
{
    if (!r) return;
    r->p   = (const uint8_t *)p;
    r->n   = p ? n : 0;
    r->off = 0;
    r->err = (!p && n) ? ENGRAM_E_ARG : ENGRAM_OK;
}

void engram_rbuf_fail(engram_rbuf *r, engram_rc rc)
{
    if (r && r->err == ENGRAM_OK && rc != ENGRAM_OK) r->err = rc;
}

/* Invariant: off <= n always, so n - off never underflows. */
static int engram_rb_have(engram_rbuf *r, size_t n)
{
    if (!r || r->err != ENGRAM_OK) return 0;
    if (n > r->n - r->off) { r->err = ENGRAM_E_FORMAT; return 0; }
    return 1;
}

int engram_rbuf_bytes(engram_rbuf *r, void *dst, size_t n)
{
    if (!n) return r && r->err == ENGRAM_OK;
    if (!dst) { engram_rbuf_fail(r, ENGRAM_E_ARG); return 0; }
    if (!engram_rb_have(r, n)) { memset(dst, 0, n); return 0; }
    memcpy(dst, r->p + r->off, n);
    r->off += n;
    return 1;
}

const uint8_t *engram_rbuf_view(engram_rbuf *r, size_t n)
{
    const uint8_t *v;
    if (!engram_rb_have(r, n)) return NULL;
    if (!n) return ENGRAM_BUF_EMPTY;
    v = r->p + r->off;
    r->off += n;
    return v;
}

int engram_rbuf_skip(engram_rbuf *r, size_t n)
{
    if (!engram_rb_have(r, n)) return 0;
    r->off += n;
    return 1;
}

uint8_t engram_rbuf_u8(engram_rbuf *r)
{
    if (!engram_rb_have(r, 1u)) return 0;
    return r->p[r->off++];
}

uint16_t engram_rbuf_u16(engram_rbuf *r)
{
    uint16_t v;
    if (!engram_rb_have(r, 2u)) return 0;
    v = (uint16_t)((uint16_t)r->p[r->off] | (uint16_t)((uint16_t)r->p[r->off + 1u] << 8));
    r->off += 2u;
    return v;
}

uint32_t engram_rbuf_u32(engram_rbuf *r)
{
    uint32_t v;
    if (!engram_rb_have(r, 4u)) return 0;
    v = engram_le_get_u32(r->p + r->off);
    r->off += 4u;
    return v;
}

uint64_t engram_rbuf_u64(engram_rbuf *r)
{
    uint64_t v;
    if (!engram_rb_have(r, 8u)) return 0;
    v = engram_le_get_u64(r->p + r->off);
    r->off += 8u;
    return v;
}

int32_t engram_rbuf_i32(engram_rbuf *r) { return (int32_t)engram_rbuf_u32(r); }
int64_t engram_rbuf_i64(engram_rbuf *r) { return (int64_t)engram_rbuf_u64(r); }

float engram_rbuf_f32(engram_rbuf *r)
{
    uint32_t u = engram_rbuf_u32(r);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

double engram_rbuf_f64(engram_rbuf *r)
{
    uint64_t u = engram_rbuf_u64(r);
    double d;
    memcpy(&d, &u, sizeof d);
    return d;
}

float engram_rbuf_f32_finite(engram_rbuf *r)
{
    uint32_t u = engram_rbuf_u32(r);
    float f;
    if (r && r->err == ENGRAM_OK && !engram_f32_bits_finite(u)) {
        engram_rbuf_fail(r, ENGRAM_E_FORMAT);
        return 0.0f;
    }
    memcpy(&f, &u, sizeof f);
    return f;
}

int engram_rbuf_f32s_finite(engram_rbuf *r, float *dst, size_t n)
{
    size_t i;
    if (!n) return r && r->err == ENGRAM_OK;
    if (!dst) { engram_rbuf_fail(r, ENGRAM_E_ARG); return 0; }
    if (n > SIZE_MAX / 4u || !engram_rb_have(r, n * 4u)) {
        engram_rbuf_fail(r, ENGRAM_E_FORMAT);
        memset(dst, 0, n * sizeof *dst);
        return 0;
    }
    for (i = 0; i < n; i++) {
        uint32_t u = engram_le_get_u32(r->p + r->off);
        if (!engram_f32_bits_finite(u)) {
            engram_rbuf_fail(r, ENGRAM_E_FORMAT);
            memset(dst, 0, n * sizeof *dst);       /* never hand back a partially-filled array */
            return 0;
        }
        memcpy(&dst[i], &u, sizeof u);
        r->off += 4u;
    }
    return 1;
}

const uint8_t *engram_rbuf_blob(engram_rbuf *r, size_t maxlen, size_t *len)
{
    uint32_t n;
    const uint8_t *v;
    if (len) *len = 0;
    n = engram_rbuf_u32(r);
    if (!r || r->err != ENGRAM_OK) return NULL;
    if ((uint64_t)n > (uint64_t)maxlen) { engram_rbuf_fail(r, ENGRAM_E_FORMAT); return NULL; }
    v = engram_rbuf_view(r, (size_t)n);
    if (v && len) *len = (size_t)n;
    return v;
}

char *engram_rbuf_str(engram_rbuf *r, size_t maxlen)
{
    size_t n;
    const uint8_t *v = engram_rbuf_blob(r, maxlen, &n);
    char *s;
    if (!v) return NULL;
    if (n && memchr(v, 0, n)) { engram_rbuf_fail(r, ENGRAM_E_FORMAT); return NULL; }
    s = (char *)engram_malloc(n + 1u);
    if (!s) { engram_rbuf_fail(r, ENGRAM_E_MEM); return NULL; }
    memcpy(s, v, n);
    s[n] = 0;
    return s;
}

size_t engram_rbuf_left(const engram_rbuf *r) { return r ? r->n - r->off : 0; }
size_t engram_rbuf_pos(const engram_rbuf *r)  { return r ? r->off : 0; }
engram_rc engram_rbuf_status(const engram_rbuf *r) { return r ? r->err : ENGRAM_E_ARG; }

engram_rc engram_rbuf_end(const engram_rbuf *r)
{
    if (!r) return ENGRAM_E_ARG;
    if (r->err != ENGRAM_OK) return r->err;
    return r->off == r->n ? ENGRAM_OK : ENGRAM_E_FORMAT;
}
