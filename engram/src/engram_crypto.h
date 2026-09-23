/* ==================================================================================================
 * engram_crypto.h -- THE CORE CRYPTOGRAPHY: every primitive the stores, the diary and the key are
 * built from, implemented from their standards and checked against the standards' own test vectors.
 * ==================================================================================================
 *
 * WHAT IS HERE, AND WHY EACH
 * ==================================================================================================
 *   SHA-256              FIPS 180-4       integrity of files; the custody chain's links (P4.4)
 *   HMAC-SHA-256         RFC 2104/4231    the custody chain's signatures; keyed integrity
 *   HKDF-SHA-256         RFC 5869         one key per file, derived from the master key and a fresh
 *                                         random salt -- so no (key, nonce) pair ever repeats
 *   BLAKE2b              RFC 7693         the hash Argon2 is built on
 *   Argon2id             RFC 9106         the password, stretched ONCE into the master key: memory-hard,
 *                                         so a stolen store cannot be brute-forced on a GPU cheaply
 *   ChaCha20-Poly1305    RFC 8439         sealing: confidentiality and authenticity in one pass,
 *                                         constant-time in software with no tables to leak through
 *                                         caches -- the reason it was chosen over AES here
 *
 * RULES THIS FILE KEEPS
 * ==================================================================================================
 *   - No secret-dependent branch or table index anywhere (ChaCha20, Poly1305, BLAKE2b, the compare).
 *     Argon2id's data-DEPENDENT addressing in its second half is the algorithm, by design.
 *   - A tag is compared in constant time, and a failed open WIPES the output before returning
 *     ENGRAM_E_AUTH: plaintext that failed authentication is never handed back.
 *   - Every secret buffer this code allocates or holds on its stack is wiped before release.
 *   - Integers only, little- or big-endian as each standard says, composed byte by byte: identical on
 *     every platform (R5).
 * ============================================================================================== */
#ifndef ENGRAM_CRYPTO_H
#define ENGRAM_CRYPTO_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- helpers --------------------------------------------------------------------------------- */
/* 1 if the n bytes are equal, 0 if not -- in time that depends on n only. */
int  engram_ct_equal(const void *a, const void *b, size_t n);
/* Zero n bytes in a way the optimiser may not remove. */
void engram_wipe(void *p, size_t n);

/* ---- SHA-256 --------------------------------------------------------------------------------- */
typedef struct {
    uint32_t h[8];
    uint64_t len;            /* bytes absorbed */
    uint8_t  buf[64];
    size_t   used;
} engram_sha256_ctx;

void engram_sha256_init(engram_sha256_ctx *c);
void engram_sha256_update(engram_sha256_ctx *c, const void *p, size_t n);
void engram_sha256_final(engram_sha256_ctx *c, uint8_t out[32]);      /* wipes the context */
void engram_sha256(const void *p, size_t n, uint8_t out[32]);

/* ---- HMAC-SHA-256 ---------------------------------------------------------------------------- */
typedef struct {
    engram_sha256_ctx in, out;
} engram_hmac_ctx;

void engram_hmac_init(engram_hmac_ctx *c, const void *key, size_t klen);
void engram_hmac_update(engram_hmac_ctx *c, const void *p, size_t n);
void engram_hmac_final(engram_hmac_ctx *c, uint8_t out[32]);          /* wipes the context */
void engram_hmac(const void *key, size_t klen, const void *p, size_t n, uint8_t out[32]);

/* ---- HKDF-SHA-256 (RFC 5869) ----------------------------------------------------------------- */
/* ENGRAM_E_ARG if okm_len > 255 * 32 or a required pointer is NULL. */
engram_rc engram_hkdf(const void *salt, size_t salt_len, const void *ikm, size_t ikm_len,
                      const void *info, size_t info_len, uint8_t *okm, size_t okm_len);

/* ---- BLAKE2b (RFC 7693) ---------------------------------------------------------------------- */
typedef struct {
    uint64_t h[8];
    uint64_t t[2];
    uint8_t  buf[128];
    size_t   used;
    size_t   outlen;
} engram_blake2b_ctx;

/* outlen 1..64, keylen 0..64 (key may be NULL when 0). ENGRAM_E_ARG otherwise. */
engram_rc engram_blake2b_init(engram_blake2b_ctx *c, size_t outlen, const void *key, size_t keylen);
void      engram_blake2b_update(engram_blake2b_ctx *c, const void *p, size_t n);
void      engram_blake2b_final(engram_blake2b_ctx *c, uint8_t *out);  /* outlen bytes; wipes */
engram_rc engram_blake2b(uint8_t *out, size_t outlen, const void *key, size_t keylen, const void *p, size_t n);

/* ---- Argon2id (RFC 9106, version 0x13) ------------------------------------------------------- */
typedef struct {
    uint32_t t_cost;         /* passes over memory, >= 1                                          */
    uint32_t m_kib;          /* memory in KiB, >= 8 * lanes                                        */
    uint32_t lanes;          /* parallelism p, 1..255 -- computed one lane after another here, so
                                the result is the standard's for p lanes, deterministically        */
} engram_argon2_params;

/* The tag of length outlen (4..1024) for the given password, salt (>= 8 bytes), optional secret and
 * associated data. Allocates m_kib KiB. ENGRAM_E_ARG on bad parameters, ENGRAM_E_MEM if the memory
 * cannot be had; the output is wiped on any failure. */
engram_rc engram_argon2id(const engram_argon2_params *prm, const void *pwd, size_t pwd_len,
                          const void *salt, size_t salt_len, const void *secret, size_t secret_len,
                          const void *ad, size_t ad_len, uint8_t *out, size_t outlen);

/* ---- ChaCha20 and Poly1305 (RFC 8439) --------------------------------------------------------- */
/* One 64-byte keystream block. */
void engram_chacha20_block(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12], uint8_t out[64]);
/* out = in XOR keystream starting at block `counter`. in and out may be the same buffer. */
void engram_chacha20_xor(const uint8_t key[32], uint32_t counter, const uint8_t nonce[12],
                         const uint8_t *in, uint8_t *out, size_t n);
typedef struct {
    uint32_t r[5], pad[4], h[5];
    uint8_t  buf[16];
    size_t   used;
} engram_poly1305_ctx;

void engram_poly1305_init(engram_poly1305_ctx *c, const uint8_t key[32]);
void engram_poly1305_update(engram_poly1305_ctx *c, const void *p, size_t n);
void engram_poly1305_final(engram_poly1305_ctx *c, uint8_t tag[16]);     /* wipes the context */
void engram_poly1305(const uint8_t key[32], const uint8_t *msg, size_t n, uint8_t tag[16]);

/* AEAD_CHACHA20_POLY1305. seal: ct (n bytes) and tag. open: ENGRAM_E_AUTH, with the output WIPED, if
 * the tag does not verify -- checked before a single byte is decrypted. in and out may alias. Neither
 * allocates, so neither can fail for want of memory. */
void      engram_aead_seal(const uint8_t key[32], const uint8_t nonce[12], const void *aad, size_t aad_len,
                           const uint8_t *pt, size_t n, uint8_t *ct, uint8_t tag[16]);
engram_rc engram_aead_open(const uint8_t key[32], const uint8_t nonce[12], const void *aad, size_t aad_len,
                           const uint8_t *ct, size_t n, const uint8_t tag[16], uint8_t *pt);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_CRYPTO_H */
