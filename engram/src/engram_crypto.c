/* ==================================================================================================
 * engram_crypto.c -- the core cryptography. Contract in engram_crypto.h; every function below follows
 * the section of its standard named beside it, and test_crypto.c checks each against the standard's
 * own vectors and against independent implementations.
 * ============================================================================================== */
#include "engram_crypto.h"
#include "engram_alloc.h"

#include <string.h>

/* ---- helpers ------------------------------------------------------------------------------------ */
int engram_ct_equal(const void *a, const void *b, size_t n)
{
    const volatile uint8_t *x = (const volatile uint8_t *)a, *y = (const volatile uint8_t *)b;
    uint8_t d = 0;
    size_t i;
    for (i = 0; i < n; i++) d = (uint8_t)(d | (x[i] ^ y[i]));
    return d == 0u;
}

void engram_wipe(void *p, size_t n)
{
    volatile uint8_t *v = (volatile uint8_t *)p;
    size_t i;
    if (!p) return;
    for (i = 0; i < n; i++) v[i] = 0u;
}

static uint32_t engram_ld32be(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}
static void engram_st32be(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v;
}
static uint32_t engram_ld32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void engram_st32le(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}
static uint64_t engram_ld64le(const uint8_t *p)
{
    return (uint64_t)engram_ld32le(p) | ((uint64_t)engram_ld32le(p + 4) << 32);
}
static void engram_st64le(uint8_t *p, uint64_t v)
{
    engram_st32le(p, (uint32_t)v);
    engram_st32le(p + 4, (uint32_t)(v >> 32));
}
static uint32_t engram_rotr32(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }
static uint32_t engram_rotl32(uint32_t x, unsigned n) { return (x << n) | (x >> (32u - n)); }
static uint64_t engram_rotr64(uint64_t x, unsigned n) { return (x >> n) | (x << (64u - n)); }

/* ---- SHA-256 (FIPS 180-4 §6.2) ------------------------------------------------------------------ */
static const uint32_t ENGRAM_SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static void engram_sha256_block(uint32_t h[8], const uint8_t *p)
{
    uint32_t w[64], a, b, c, d, e, f, g, hh;
    unsigned i;
    for (i = 0; i < 16u; i++) w[i] = engram_ld32be(p + 4u * i);
    for (i = 16; i < 64u; i++) {
        uint32_t s0 = engram_rotr32(w[i - 15u], 7) ^ engram_rotr32(w[i - 15u], 18) ^ (w[i - 15u] >> 3);
        uint32_t s1 = engram_rotr32(w[i - 2u], 17) ^ engram_rotr32(w[i - 2u], 19) ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = h[0]; b = h[1]; c = h[2]; d = h[3]; e = h[4]; f = h[5]; g = h[6]; hh = h[7];
    for (i = 0; i < 64u; i++) {
        uint32_t S1 = engram_rotr32(e, 6) ^ engram_rotr32(e, 11) ^ engram_rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + ENGRAM_SHA256_K[i] + w[i];
        uint32_t S0 = engram_rotr32(a, 2) ^ engram_rotr32(a, 13) ^ engram_rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    engram_wipe(w, sizeof w);
}

void engram_sha256_init(engram_sha256_ctx *c)
{
    static const uint32_t IV[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    memcpy(c->h, IV, sizeof IV);
    c->len = 0;
    c->used = 0;
}

void engram_sha256_update(engram_sha256_ctx *c, const void *p, size_t n)
{
    const uint8_t *s = (const uint8_t *)p;
    c->len += (uint64_t)n;
    while (n > 0u) {
        size_t take = 64u - c->used;
        if (take > n) take = n;
        memcpy(c->buf + c->used, s, take);
        c->used += take; s += take; n -= take;
        if (c->used == 64u) { engram_sha256_block(c->h, c->buf); c->used = 0; }
    }
}

void engram_sha256_final(engram_sha256_ctx *c, uint8_t out[32])
{
    uint64_t bits = c->len * 8u;
    unsigned i;
    c->buf[c->used++] = 0x80u;
    if (c->used > 56u) {
        memset(c->buf + c->used, 0, 64u - c->used);
        engram_sha256_block(c->h, c->buf);
        c->used = 0;
    }
    memset(c->buf + c->used, 0, 56u - c->used);
    for (i = 0; i < 8u; i++) c->buf[56u + i] = (uint8_t)(bits >> (56u - 8u * i));
    engram_sha256_block(c->h, c->buf);
    for (i = 0; i < 8u; i++) engram_st32be(out + 4u * i, c->h[i]);
    engram_wipe(c, sizeof *c);
}

void engram_sha256(const void *p, size_t n, uint8_t out[32])
{
    engram_sha256_ctx c;
    engram_sha256_init(&c);
    engram_sha256_update(&c, p, n);
    engram_sha256_final(&c, out);
}

/* ---- HMAC-SHA-256 (RFC 2104) -------------------------------------------------------------------- */
void engram_hmac_init(engram_hmac_ctx *c, const void *key, size_t klen)
{
    uint8_t k[64], pad[64];
    unsigned i;
    memset(k, 0, sizeof k);
    if (klen > 64u) engram_sha256(key, klen, k);
    else if (klen) memcpy(k, key, klen);
    for (i = 0; i < 64u; i++) pad[i] = (uint8_t)(k[i] ^ 0x36u);
    engram_sha256_init(&c->in);
    engram_sha256_update(&c->in, pad, 64u);
    for (i = 0; i < 64u; i++) pad[i] = (uint8_t)(k[i] ^ 0x5cu);
    engram_sha256_init(&c->out);
    engram_sha256_update(&c->out, pad, 64u);
    engram_wipe(k, sizeof k);
    engram_wipe(pad, sizeof pad);
}

void engram_hmac_update(engram_hmac_ctx *c, const void *p, size_t n) { engram_sha256_update(&c->in, p, n); }

void engram_hmac_final(engram_hmac_ctx *c, uint8_t out[32])
{
    uint8_t inner[32];
    engram_sha256_final(&c->in, inner);
    engram_sha256_update(&c->out, inner, 32u);
    engram_sha256_final(&c->out, out);
    engram_wipe(inner, sizeof inner);
    engram_wipe(c, sizeof *c);
}

void engram_hmac(const void *key, size_t klen, const void *p, size_t n, uint8_t out[32])
{
    engram_hmac_ctx c;
    engram_hmac_init(&c, key, klen);
    engram_hmac_update(&c, p, n);
    engram_hmac_final(&c, out);
}

/* ---- HKDF-SHA-256 (RFC 5869 §2) ------------------------------------------------------------------ */
engram_rc engram_hkdf(const void *salt, size_t salt_len, const void *ikm, size_t ikm_len,
                      const void *info, size_t info_len, uint8_t *okm, size_t okm_len)
{
    uint8_t prk[32], t[32], zeros[32];
    size_t done = 0, tlen = 0;
    unsigned counter = 1;
    if ((!salt && salt_len) || (!ikm && ikm_len) || (!info && info_len) || (!okm && okm_len)) return ENGRAM_E_ARG;
    if (okm_len > 255u * 32u) return ENGRAM_E_ARG;
    memset(zeros, 0, sizeof zeros);
    /* extract: an absent salt is HashLen zeros */
    engram_hmac(salt_len ? salt : zeros, salt_len ? salt_len : 32u, ikm, ikm_len, prk);
    /* expand: T(i) = HMAC(PRK, T(i-1) | info | i) */
    while (done < okm_len) {
        engram_hmac_ctx c;
        uint8_t ib = (uint8_t)counter;
        size_t take;
        engram_hmac_init(&c, prk, 32u);
        engram_hmac_update(&c, t, tlen);
        engram_hmac_update(&c, info, info_len);
        engram_hmac_update(&c, &ib, 1u);
        engram_hmac_final(&c, t);
        tlen = 32u;
        take = okm_len - done < 32u ? okm_len - done : 32u;
        memcpy(okm + done, t, take);
        done += take;
        counter++;
    }
    engram_wipe(prk, sizeof prk);
    engram_wipe(t, sizeof t);
    return ENGRAM_OK;
}

/* ---- BLAKE2b (RFC 7693 §3) ------------------------------------------------------------------------ */
static const uint64_t ENGRAM_B2B_IV[8] = {
    0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull, 0xa54ff53a5f1d36f1ull,
    0x510e527fade682d1ull, 0x9b05688c2b3e6c1full, 0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull
};
static const uint8_t ENGRAM_B2B_SIGMA[12][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
    { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
    { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
    { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
    { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
    { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
    { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
    { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
    { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
    { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
};

#define ENGRAM_B2B_G(v, a, b, c, d, x, y)                                            \
    do {                                                                             \
        v[a] = v[a] + v[b] + (x); v[d] = engram_rotr64(v[d] ^ v[a], 32);             \
        v[c] = v[c] + v[d];       v[b] = engram_rotr64(v[b] ^ v[c], 24);             \
        v[a] = v[a] + v[b] + (y); v[d] = engram_rotr64(v[d] ^ v[a], 16);             \
        v[c] = v[c] + v[d];       v[b] = engram_rotr64(v[b] ^ v[c], 63);             \
    } while (0)

static void engram_blake2b_compress(engram_blake2b_ctx *c, const uint8_t *block, int last)
{
    uint64_t v[16], m[16];
    unsigned i, r;
    for (i = 0; i < 16u; i++) m[i] = engram_ld64le(block + 8u * i);
    for (i = 0; i < 8u; i++) { v[i] = c->h[i]; v[i + 8u] = ENGRAM_B2B_IV[i]; }
    v[12] ^= c->t[0];
    v[13] ^= c->t[1];
    if (last) v[14] = ~v[14];
    for (r = 0; r < 12u; r++) {
        const uint8_t *s = ENGRAM_B2B_SIGMA[r];
        ENGRAM_B2B_G(v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        ENGRAM_B2B_G(v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        ENGRAM_B2B_G(v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        ENGRAM_B2B_G(v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        ENGRAM_B2B_G(v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        ENGRAM_B2B_G(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        ENGRAM_B2B_G(v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        ENGRAM_B2B_G(v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for (i = 0; i < 8u; i++) c->h[i] ^= v[i] ^ v[i + 8u];
    engram_wipe(v, sizeof v);
    engram_wipe(m, sizeof m);
}

engram_rc engram_blake2b_init(engram_blake2b_ctx *c, size_t outlen, const void *key, size_t keylen)
{
    unsigned i;
    if (!c || outlen == 0u || outlen > 64u || keylen > 64u || (!key && keylen)) return ENGRAM_E_ARG;
    for (i = 0; i < 8u; i++) c->h[i] = ENGRAM_B2B_IV[i];
    c->h[0] ^= 0x01010000ull ^ ((uint64_t)keylen << 8) ^ (uint64_t)outlen;
    c->t[0] = 0; c->t[1] = 0;
    c->used = 0;
    c->outlen = outlen;
    memset(c->buf, 0, sizeof c->buf);
    if (keylen) {                                        /* the key is the first block, zero-padded */
        memcpy(c->buf, key, keylen);
        c->used = 128u;
    }
    return ENGRAM_OK;
}

void engram_blake2b_update(engram_blake2b_ctx *c, const void *p, size_t n)
{
    const uint8_t *s = (const uint8_t *)p;
    while (n > 0u) {
        size_t take;
        if (c->used == 128u) {                           /* compress only once more input is known */
            c->t[0] += 128u;
            if (c->t[0] < 128u) c->t[1]++;
            engram_blake2b_compress(c, c->buf, 0);
            c->used = 0;
        }
        take = 128u - c->used;
        if (take > n) take = n;
        memcpy(c->buf + c->used, s, take);
        c->used += take; s += take; n -= take;
    }
}

void engram_blake2b_final(engram_blake2b_ctx *c, uint8_t *out)
{
    uint8_t full[64];
    unsigned i;
    c->t[0] += (uint64_t)c->used;
    if (c->t[0] < (uint64_t)c->used) c->t[1]++;
    memset(c->buf + c->used, 0, 128u - c->used);
    engram_blake2b_compress(c, c->buf, 1);
    for (i = 0; i < 8u; i++) engram_st64le(full + 8u * i, c->h[i]);
    memcpy(out, full, c->outlen);
    engram_wipe(full, sizeof full);
    engram_wipe(c, sizeof *c);
}

engram_rc engram_blake2b(uint8_t *out, size_t outlen, const void *key, size_t keylen, const void *p, size_t n)
{
    engram_blake2b_ctx c;
    engram_rc rc;
    if (!out || (!p && n)) return ENGRAM_E_ARG;
    rc = engram_blake2b_init(&c, outlen, key, keylen);
    if (rc != ENGRAM_OK) return rc;
    engram_blake2b_update(&c, p, n);
    engram_blake2b_final(&c, out);
    return ENGRAM_OK;
}

/* ---- Argon2id (RFC 9106 §3) ------------------------------------------------------------------------ */
#define ENGRAM_A2_BLOCK 1024u
#define ENGRAM_A2_QW    128u                              /* 64-bit words per block */
#define ENGRAM_A2_SYNC  4u                                /* slices per pass */

typedef struct { uint64_t v[ENGRAM_A2_QW]; } engram_a2_block;

/* H' (§3.3): variable-length hash built from BLAKE2b */
static void engram_a2_hprime(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen)
{
    engram_blake2b_ctx c;
    uint8_t len4[4], v[64];
    engram_st32le(len4, (uint32_t)outlen);
    if (outlen <= 64u) {
        (void)engram_blake2b_init(&c, outlen, NULL, 0);
        engram_blake2b_update(&c, len4, 4u);
        engram_blake2b_update(&c, in, inlen);
        engram_blake2b_final(&c, out);
        return;
    }
    {
        size_t r = (outlen + 31u) / 32u - 2u, i, done;
        (void)engram_blake2b_init(&c, 64u, NULL, 0);
        engram_blake2b_update(&c, len4, 4u);
        engram_blake2b_update(&c, in, inlen);
        engram_blake2b_final(&c, v);
        memcpy(out, v, 32u);
        done = 32u;
        for (i = 1; i < r; i++) {
            (void)engram_blake2b(v, 64u, NULL, 0, v, 64u);
            memcpy(out + done, v, 32u);
            done += 32u;
        }
        (void)engram_blake2b(v, outlen - 32u * r, NULL, 0, v, 64u);
        memcpy(out + done, v, outlen - 32u * r);
        engram_wipe(v, sizeof v);
    }
}

/* GB (§3.6): BLAKE2b's G with the multiplication 2 * lo32(a) * lo32(b) added in */
static uint64_t engram_a2_fbla(uint64_t x, uint64_t y)
{
    return x + y + 2u * (uint64_t)(uint32_t)x * (uint64_t)(uint32_t)y;
}

#define ENGRAM_A2_GB(a, b, c, d)                                                     \
    do {                                                                             \
        a = engram_a2_fbla(a, b); d = engram_rotr64(d ^ a, 32);                      \
        c = engram_a2_fbla(c, d); b = engram_rotr64(b ^ c, 24);                      \
        a = engram_a2_fbla(a, b); d = engram_rotr64(d ^ a, 16);                      \
        c = engram_a2_fbla(c, d); b = engram_rotr64(b ^ c, 63);                      \
    } while (0)

/* P on 16 words given by index */
static void engram_a2_p(uint64_t *v, unsigned i0, unsigned i1, unsigned i2, unsigned i3, unsigned i4,
                        unsigned i5, unsigned i6, unsigned i7, unsigned i8, unsigned i9, unsigned i10,
                        unsigned i11, unsigned i12, unsigned i13, unsigned i14, unsigned i15)
{
    ENGRAM_A2_GB(v[i0], v[i4], v[i8], v[i12]);
    ENGRAM_A2_GB(v[i1], v[i5], v[i9], v[i13]);
    ENGRAM_A2_GB(v[i2], v[i6], v[i10], v[i14]);
    ENGRAM_A2_GB(v[i3], v[i7], v[i11], v[i15]);
    ENGRAM_A2_GB(v[i0], v[i5], v[i10], v[i15]);
    ENGRAM_A2_GB(v[i1], v[i6], v[i11], v[i12]);
    ENGRAM_A2_GB(v[i2], v[i7], v[i8], v[i13]);
    ENGRAM_A2_GB(v[i3], v[i4], v[i9], v[i14]);
}

/* G (§3.5): out = (with_xor ? out : 0) XOR P(X XOR Y) XOR X XOR Y, where P runs over the 8 rows and then
 * the 8 columns of the 8x8 matrix of 16-byte registers */
static void engram_a2_g(const engram_a2_block *x, const engram_a2_block *y, engram_a2_block *out, int with_xor)
{
    engram_a2_block r, z;
    unsigned i;
    for (i = 0; i < ENGRAM_A2_QW; i++) r.v[i] = x->v[i] ^ y->v[i];
    z = r;
    for (i = 0; i < 8u; i++) {                           /* rows: 16 consecutive words each */
        unsigned b = 16u * i;
        engram_a2_p(z.v, b, b + 1u, b + 2u, b + 3u, b + 4u, b + 5u, b + 6u, b + 7u,
                    b + 8u, b + 9u, b + 10u, b + 11u, b + 12u, b + 13u, b + 14u, b + 15u);
    }
    for (i = 0; i < 8u; i++) {                           /* columns: word pairs 2i, 2i+1 of every row */
        unsigned b = 2u * i;
        engram_a2_p(z.v, b, b + 1u, b + 16u, b + 17u, b + 32u, b + 33u, b + 48u, b + 49u,
                    b + 64u, b + 65u, b + 80u, b + 81u, b + 96u, b + 97u, b + 112u, b + 113u);
    }
    for (i = 0; i < ENGRAM_A2_QW; i++) {
        uint64_t v = z.v[i] ^ r.v[i];
        out->v[i] = with_xor ? out->v[i] ^ v : v;
    }
    engram_wipe(&r, sizeof r);
    engram_wipe(&z, sizeof z);
}

static void engram_a2_store(uint8_t *p, const engram_a2_block *b)
{
    unsigned i;
    for (i = 0; i < ENGRAM_A2_QW; i++) engram_st64le(p + 8u * i, b->v[i]);
}

static void engram_a2_load(engram_a2_block *b, const uint8_t *p)
{
    unsigned i;
    for (i = 0; i < ENGRAM_A2_QW; i++) b->v[i] = engram_ld64le(p + 8u * i);
}

engram_rc engram_argon2id(const engram_argon2_params *prm, const void *pwd, size_t pwd_len,
                          const void *salt, size_t salt_len, const void *secret, size_t secret_len,
                          const void *ad, size_t ad_len, uint8_t *out, size_t outlen)
{
    engram_blake2b_ctx hc;
    uint8_t h0[72], le[4], blockbytes[ENGRAM_A2_BLOCK];
    engram_a2_block *B = NULL, addr, zero, inputb;
    uint32_t p, m_blocks, q, segment, pass, slice, lane, idx;
    size_t total;
    if (!prm || !out || outlen < 4u || outlen > 1024u) return ENGRAM_E_ARG;
    if ((!pwd && pwd_len) || !salt || salt_len < 8u || (!secret && secret_len) || (!ad && ad_len)) {
        engram_wipe(out, outlen);
        return ENGRAM_E_ARG;
    }
    p = prm->lanes;
    if (p < 1u || p > 255u || prm->t_cost < 1u || prm->m_kib < 8u * p ||
        pwd_len > 0xFFFFFFFFu || salt_len > 0xFFFFFFFFu || secret_len > 0xFFFFFFFFu || ad_len > 0xFFFFFFFFu) {
        engram_wipe(out, outlen);
        return ENGRAM_E_ARG;
    }
    /* m' = 4p * floor(m / 4p) blocks; q columns per lane, 4 segments per lane per pass */
    m_blocks = 4u * p * (prm->m_kib / (4u * p));
    q = m_blocks / p;
    segment = q / ENGRAM_A2_SYNC;
    total = (size_t)m_blocks;
    B = (engram_a2_block *)engram_array(total, sizeof *B);
    if (!B) { engram_wipe(out, outlen); return ENGRAM_E_MEM; }

    /* H0 (§3.2) = BLAKE2b-512(p, T, m, t, v, y, |P|, P, |S|, S, |K|, K, |X|, X) */
    (void)engram_blake2b_init(&hc, 64u, NULL, 0);
#define A2_U32(x) do { engram_st32le(le, (uint32_t)(x)); engram_blake2b_update(&hc, le, 4u); } while (0)
    A2_U32(p); A2_U32(outlen); A2_U32(prm->m_kib); A2_U32(prm->t_cost); A2_U32(0x13u); A2_U32(2u);
    A2_U32(pwd_len); engram_blake2b_update(&hc, pwd, pwd_len);
    A2_U32(salt_len); engram_blake2b_update(&hc, salt, salt_len);
    A2_U32(secret_len); engram_blake2b_update(&hc, secret, secret_len);
    A2_U32(ad_len); engram_blake2b_update(&hc, ad, ad_len);
#undef A2_U32
    engram_blake2b_final(&hc, h0);

    /* the first two blocks of every lane (§3.4) */
    for (lane = 0; lane < p; lane++) {
        unsigned j;
        for (j = 0; j < 2u; j++) {
            engram_st32le(h0 + 64, j);
            engram_st32le(h0 + 68, lane);
            engram_a2_hprime(blockbytes, ENGRAM_A2_BLOCK, h0, 72u);
            engram_a2_load(&B[(size_t)lane * q + j], blockbytes);
        }
    }
    memset(&zero, 0, sizeof zero);

    for (pass = 0; pass < prm->t_cost; pass++)
        for (slice = 0; slice < ENGRAM_A2_SYNC; slice++)
            for (lane = 0; lane < p; lane++) {
                /* Argon2id: data-independent addressing in the first half of the first pass */
                int indep = pass == 0u && slice < 2u;
                uint32_t start = (pass == 0u && slice == 0u) ? 2u : 0u, counter = 0;
                if (indep) {
                    memset(&inputb, 0, sizeof inputb);
                    inputb.v[0] = pass; inputb.v[1] = lane; inputb.v[2] = slice;
                    inputb.v[3] = m_blocks; inputb.v[4] = prm->t_cost; inputb.v[5] = 2u;
                }
                for (idx = start; idx < segment; idx++) {
                    uint32_t col = slice * segment + idx, prev, ref_lane, ref_area, rel, ref_col;
                    uint64_t pseudo, x, y, zz;
                    size_t cur_i = (size_t)lane * q + col;
                    prev = col == 0u ? q - 1u : col - 1u;
                    if (indep) {
                        if (idx == start || idx % ENGRAM_A2_QW == 0u) {
                            /* the next address block: G(0, G(0, input)) with the counter bumped */
                            inputb.v[6] = ++counter;
                            engram_a2_g(&zero, &inputb, &addr, 0);
                            engram_a2_g(&zero, &addr, &addr, 0);
                        }
                        pseudo = addr.v[idx % ENGRAM_A2_QW];
                    } else pseudo = B[(size_t)lane * q + prev].v[0];
                    /* §3.4.1.2: which lane, then which block of the reference set */
                    ref_lane = (pass == 0u && slice == 0u) ? lane : (uint32_t)((pseudo >> 32) % p);
                    if (pass == 0u) {
                        if (ref_lane == lane) ref_area = slice * segment + idx - 1u;
                        else ref_area = slice * segment + (idx == 0u ? (uint32_t)-1 : 0u);
                    } else {
                        if (ref_lane == lane) ref_area = q - segment + idx - 1u;
                        else ref_area = q - segment + (idx == 0u ? (uint32_t)-1 : 0u);
                    }
                    x = (uint64_t)(uint32_t)pseudo;
                    y = (x * x) >> 32;
                    zz = (uint64_t)ref_area - 1u - (((uint64_t)ref_area * y) >> 32);
                    rel = (uint32_t)zz;
                    /* the reference set starts at column 0 in the first pass, else just after this segment */
                    ref_col = pass == 0u ? rel
                                         : (uint32_t)((((uint64_t)(slice + 1u) % ENGRAM_A2_SYNC) * segment + rel) % q);
                    engram_a2_g(&B[(size_t)lane * q + prev], &B[(size_t)ref_lane * q + ref_col], &B[cur_i],
                                pass > 0u);
                }
            }

    /* the final block: XOR of every lane's last column, then H' (§3.4) */
    {
        engram_a2_block fin = B[q - 1u];
        uint32_t i;
        for (lane = 1; lane < p; lane++)
            for (i = 0; i < ENGRAM_A2_QW; i++) fin.v[i] ^= B[(size_t)lane * q + q - 1u].v[i];
        engram_a2_store(blockbytes, &fin);
        engram_a2_hprime(out, outlen, blockbytes, ENGRAM_A2_BLOCK);
        engram_wipe(&fin, sizeof fin);
    }
    engram_wipe(B, total * sizeof *B);
    engram_free(B);
    engram_wipe(h0, sizeof h0);
    engram_wipe(blockbytes, sizeof blockbytes);
    engram_wipe(&addr, sizeof addr);
    engram_wipe(&inputb, sizeof inputb);
    return ENGRAM_OK;
}

/* ---- ChaCha20 (RFC 8439 §2.3, §2.4) ------------------------------------------------------------------ */
#define ENGRAM_QR(a, b, c, d)                                                        \
    do {                                                                             \
        a += b; d ^= a; d = engram_rotl32(d, 16);                                    \
        c += d; b ^= c; b = engram_rotl32(b, 12);                                    \
        a += b; d ^= a; d = engram_rotl32(d, 8);                                     \
        c += d; b ^= c; b = engram_rotl32(b, 7);                                     \
    } while (0)

void engram_chacha20_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64])
{
    uint32_t s[16], x[16];
    unsigned i;
    s[0] = 0x61707865u; s[1] = 0x3320646eu; s[2] = 0x79622d32u; s[3] = 0x6b206574u;
    for (i = 0; i < 8u; i++) s[4u + i] = engram_ld32le(key + 4u * i);
    s[12] = counter;
    for (i = 0; i < 3u; i++) s[13u + i] = engram_ld32le(nonce + 4u * i);
    memcpy(x, s, sizeof s);
    for (i = 0; i < 10u; i++) {
        ENGRAM_QR(x[0], x[4], x[8], x[12]); ENGRAM_QR(x[1], x[5], x[9], x[13]);
        ENGRAM_QR(x[2], x[6], x[10], x[14]); ENGRAM_QR(x[3], x[7], x[11], x[15]);
        ENGRAM_QR(x[0], x[5], x[10], x[15]); ENGRAM_QR(x[1], x[6], x[11], x[12]);
        ENGRAM_QR(x[2], x[7], x[8], x[13]); ENGRAM_QR(x[3], x[4], x[9], x[14]);
    }
    for (i = 0; i < 16u; i++) engram_st32le(out + 4u * i, x[i] + s[i]);
    engram_wipe(s, sizeof s);
    engram_wipe(x, sizeof x);
}

void engram_chacha20_xor(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12],
                         const uint8_t *in, uint8_t *out, size_t n)
{
    uint8_t ks[64];
    size_t off = 0, i;
    while (off < n) {
        size_t take = n - off < 64u ? n - off : 64u;
        engram_chacha20_block(key, counter++, nonce, ks);
        for (i = 0; i < take; i++) out[off + i] = (uint8_t)(in[off + i] ^ ks[i]);
        off += take;
    }
    engram_wipe(ks, sizeof ks);
}

/* ---- Poly1305 (RFC 8439 §2.5) in five 26-bit limbs: 64-bit products only, portable C99 ------------ */
static void engram_poly1305_blocks(engram_poly1305_ctx *c, const uint8_t *m, size_t n, uint32_t hibit)
{
    uint32_t r0 = c->r[0], r1 = c->r[1], r2 = c->r[2], r3 = c->r[3], r4 = c->r[4];
    uint32_t s1 = r1 * 5u, s2 = r2 * 5u, s3 = r3 * 5u, s4 = r4 * 5u;
    uint32_t h0 = c->h[0], h1 = c->h[1], h2 = c->h[2], h3 = c->h[3], h4 = c->h[4], cc;
    uint64_t d0, d1, d2, d3, d4;
    while (n >= 16u) {
        h0 += (engram_ld32le(m + 0)) & 0x3ffffffu;
        h1 += (engram_ld32le(m + 3) >> 2) & 0x3ffffffu;
        h2 += (engram_ld32le(m + 6) >> 4) & 0x3ffffffu;
        h3 += (engram_ld32le(m + 9) >> 6) & 0x3ffffffu;
        h4 += (engram_ld32le(m + 12) >> 8) | hibit;
        d0 = (uint64_t)h0 * r0 + (uint64_t)h1 * s4 + (uint64_t)h2 * s3 + (uint64_t)h3 * s2 + (uint64_t)h4 * s1;
        d1 = (uint64_t)h0 * r1 + (uint64_t)h1 * r0 + (uint64_t)h2 * s4 + (uint64_t)h3 * s3 + (uint64_t)h4 * s2;
        d2 = (uint64_t)h0 * r2 + (uint64_t)h1 * r1 + (uint64_t)h2 * r0 + (uint64_t)h3 * s4 + (uint64_t)h4 * s3;
        d3 = (uint64_t)h0 * r3 + (uint64_t)h1 * r2 + (uint64_t)h2 * r1 + (uint64_t)h3 * r0 + (uint64_t)h4 * s4;
        d4 = (uint64_t)h0 * r4 + (uint64_t)h1 * r3 + (uint64_t)h2 * r2 + (uint64_t)h3 * r1 + (uint64_t)h4 * r0;
        cc = (uint32_t)(d0 >> 26); h0 = (uint32_t)d0 & 0x3ffffffu;
        d1 += cc; cc = (uint32_t)(d1 >> 26); h1 = (uint32_t)d1 & 0x3ffffffu;
        d2 += cc; cc = (uint32_t)(d2 >> 26); h2 = (uint32_t)d2 & 0x3ffffffu;
        d3 += cc; cc = (uint32_t)(d3 >> 26); h3 = (uint32_t)d3 & 0x3ffffffu;
        d4 += cc; cc = (uint32_t)(d4 >> 26); h4 = (uint32_t)d4 & 0x3ffffffu;
        h0 += cc * 5u; cc = h0 >> 26; h0 &= 0x3ffffffu;
        h1 += cc;
        m += 16; n -= 16u;
    }
    c->h[0] = h0; c->h[1] = h1; c->h[2] = h2; c->h[3] = h3; c->h[4] = h4;
}

void engram_poly1305_init(engram_poly1305_ctx *c, const uint8_t key[32])
{
    c->r[0] = (engram_ld32le(key + 0)) & 0x3ffffffu;             /* r, clamped (§2.5.1) */
    c->r[1] = (engram_ld32le(key + 3) >> 2) & 0x3ffff03u;
    c->r[2] = (engram_ld32le(key + 6) >> 4) & 0x3ffc0ffu;
    c->r[3] = (engram_ld32le(key + 9) >> 6) & 0x3f03fffu;
    c->r[4] = (engram_ld32le(key + 12) >> 8) & 0x00fffffu;
    c->pad[0] = engram_ld32le(key + 16); c->pad[1] = engram_ld32le(key + 20);
    c->pad[2] = engram_ld32le(key + 24); c->pad[3] = engram_ld32le(key + 28);
    c->h[0] = c->h[1] = c->h[2] = c->h[3] = c->h[4] = 0u;
    c->used = 0;
}

void engram_poly1305_update(engram_poly1305_ctx *c, const void *p, size_t n)
{
    const uint8_t *m = (const uint8_t *)p;
    if (c->used) {                                       /* finish a pending partial block */
        size_t take = 16u - c->used;
        if (take > n) take = n;
        memcpy(c->buf + c->used, m, take);
        c->used += take; m += take; n -= take;
        if (c->used < 16u) return;
        engram_poly1305_blocks(c, c->buf, 16u, 1u << 24);
        c->used = 0;
    }
    if (n >= 16u) {
        size_t whole = n & ~(size_t)15u;
        engram_poly1305_blocks(c, m, whole, 1u << 24);
        m += whole; n -= whole;
    }
    if (n) { memcpy(c->buf, m, n); c->used = n; }
}

void engram_poly1305_final(engram_poly1305_ctx *c, uint8_t tag[16])
{
    uint32_t h0, h1, h2, h3, h4, cc, g0, g1, g2, g3, g4, mask;
    uint64_t f;
    if (c->used) {                                       /* a short last block: 0x01 then zeros, no high bit */
        c->buf[c->used] = 1u;
        memset(c->buf + c->used + 1u, 0, 16u - c->used - 1u);
        engram_poly1305_blocks(c, c->buf, 16u, 0u);
    }
    h0 = c->h[0]; h1 = c->h[1]; h2 = c->h[2]; h3 = c->h[3]; h4 = c->h[4];
    /* full carry, then h mod 2^130 - 5, the choice made in constant time */
    cc = h1 >> 26; h1 &= 0x3ffffffu;
    h2 += cc; cc = h2 >> 26; h2 &= 0x3ffffffu;
    h3 += cc; cc = h3 >> 26; h3 &= 0x3ffffffu;
    h4 += cc; cc = h4 >> 26; h4 &= 0x3ffffffu;
    h0 += cc * 5u; cc = h0 >> 26; h0 &= 0x3ffffffu;
    h1 += cc;
    g0 = h0 + 5u; cc = g0 >> 26; g0 &= 0x3ffffffu;
    g1 = h1 + cc; cc = g1 >> 26; g1 &= 0x3ffffffu;
    g2 = h2 + cc; cc = g2 >> 26; g2 &= 0x3ffffffu;
    g3 = h3 + cc; cc = g3 >> 26; g3 &= 0x3ffffffu;
    g4 = h4 + cc - (1u << 26);
    mask = (g4 >> 31) - 1u;                              /* all ones when h >= p: take g */
    g0 &= mask; g1 &= mask; g2 &= mask; g3 &= mask; g4 &= mask;
    mask = ~mask;
    h0 = (h0 & mask) | g0; h1 = (h1 & mask) | g1; h2 = (h2 & mask) | g2; h3 = (h3 & mask) | g3; h4 = (h4 & mask) | g4;
    h0 = h0 | (h1 << 26);                                /* to 128 bits, then + s */
    h1 = (h1 >> 6) | (h2 << 20);
    h2 = (h2 >> 12) | (h3 << 14);
    h3 = (h3 >> 18) | (h4 << 8);
    f = (uint64_t)h0 + c->pad[0];             h0 = (uint32_t)f;
    f = (uint64_t)h1 + c->pad[1] + (f >> 32); h1 = (uint32_t)f;
    f = (uint64_t)h2 + c->pad[2] + (f >> 32); h2 = (uint32_t)f;
    f = (uint64_t)h3 + c->pad[3] + (f >> 32); h3 = (uint32_t)f;
    engram_st32le(tag + 0, h0); engram_st32le(tag + 4, h1); engram_st32le(tag + 8, h2); engram_st32le(tag + 12, h3);
    engram_wipe(c, sizeof *c);
}

void engram_poly1305(const uint8_t key[32], const uint8_t *msg, size_t n, uint8_t tag[16])
{
    engram_poly1305_ctx c;
    engram_poly1305_init(&c, key);
    engram_poly1305_update(&c, msg, n);
    engram_poly1305_final(&c, tag);
}

/* ---- AEAD_CHACHA20_POLY1305 (RFC 8439 §2.8): streamed, so it allocates nothing and cannot fail ------ */
static void engram_aead_tag(const uint8_t key[32], const uint8_t nonce[12], const void *aad, size_t aad_len,
                            const uint8_t *ct, size_t n, uint8_t tag[16])
{
    static const uint8_t zeros[16] = { 0 };
    uint8_t otk[64], lens[16];
    engram_poly1305_ctx pc;
    engram_chacha20_block(key, 0u, nonce, otk);          /* the one-time Poly1305 key (§2.6) */
    engram_poly1305_init(&pc, otk);
    engram_poly1305_update(&pc, aad, aad_len);
    engram_poly1305_update(&pc, zeros, (16u - aad_len % 16u) % 16u);
    engram_poly1305_update(&pc, ct, n);
    engram_poly1305_update(&pc, zeros, (16u - n % 16u) % 16u);
    engram_st64le(lens, (uint64_t)aad_len);
    engram_st64le(lens + 8, (uint64_t)n);
    engram_poly1305_update(&pc, lens, 16u);
    engram_poly1305_final(&pc, tag);
    engram_wipe(otk, sizeof otk);
}

void engram_aead_seal(const uint8_t key[32], const uint8_t nonce[12], const void *aad, size_t aad_len,
                      const uint8_t *pt, size_t n, uint8_t *ct, uint8_t tag[16])
{
    engram_chacha20_xor(key, 1u, nonce, pt, ct, n);
    engram_aead_tag(key, nonce, aad, aad_len, ct, n, tag);
}

engram_rc engram_aead_open(const uint8_t key[32], const uint8_t nonce[12], const void *aad, size_t aad_len,
                           const uint8_t *ct, size_t n, const uint8_t tag[16], uint8_t *pt)
{
    uint8_t want[16];
    engram_aead_tag(key, nonce, aad, aad_len, ct, n, want);
    if (!engram_ct_equal(want, tag, 16u)) {
        engram_wipe(want, sizeof want);
        if (pt && pt != ct) engram_wipe(pt, n);
        return ENGRAM_E_AUTH;
    }
    engram_wipe(want, sizeof want);
    engram_chacha20_xor(key, 1u, nonce, ct, pt, n);
    return ENGRAM_OK;
}
