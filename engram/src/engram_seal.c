/* ==================================================================================================
 * engram_seal.c -- the container and the keyfile. Contract in engram_seal.h.
 * ============================================================================================== */
#include "engram_seal.h"
#include "engram_alloc.h"
#include "engram_buf.h"
#include "engram_plat.h"

#include <string.h>

static const uint8_t ENGRAM_SEAL_MAGIC[8] = { 'E', 'N', 'G', 'R', 'A', 'M', 0x1A, 0x0A };
static const char ENGRAM_KEYCHECK[] = "ENGRAM key check";

void engram_key_wipe(engram_key *key)
{
    if (key) engram_wipe(key->k, sizeof key->k);
}

/* the per-write file key and nonce: HKDF(master, salt, "ENGRAM seal v1" | le16(kind)) */
static void engram_seal_derive(const engram_key *key, const uint8_t *salt, uint16_t kind, uint8_t okm[44])
{
    uint8_t info[16];
    memcpy(info, "ENGRAM seal v1", 14u);
    info[14] = (uint8_t)kind;
    info[15] = (uint8_t)(kind >> 8);
    (void)engram_hkdf(salt, 32u, key->k, sizeof key->k, info, sizeof info, okm, 44u);
    engram_wipe(info, sizeof info);
}

engram_rc engram_seal_pack(uint16_t kind, const engram_key *key, const uint8_t *salt, const void *payload,
                           size_t n, uint8_t **out, size_t *out_n)
{
    uint8_t *buf, s[32];
    size_t total;
    engram_rc rc;
    if (out) *out = NULL;
    if (out_n) *out_n = 0;
    if (!out || !out_n || (!payload && n) || kind == 0u) return ENGRAM_E_ARG;
    if (n > ENGRAM_FILE_MAX - ENGRAM_SEAL_HEADER - ENGRAM_SEAL_TRAILER) return ENGRAM_E_OVERFLOW;
    total = ENGRAM_SEAL_HEADER + n + ENGRAM_SEAL_TRAILER;
    if (salt) memcpy(s, salt, 32u);
    else if ((rc = engram_os_random(s, sizeof s)) != ENGRAM_OK) return rc;
    buf = (uint8_t *)engram_malloc(total);
    if (!buf) return ENGRAM_E_MEM;
    memset(buf, 0, ENGRAM_SEAL_HEADER);
    memcpy(buf, ENGRAM_SEAL_MAGIC, 8u);
    buf[8] = (uint8_t)ENGRAM_SEAL_VERSION; buf[9] = (uint8_t)(ENGRAM_SEAL_VERSION >> 8);
    buf[10] = (uint8_t)kind; buf[11] = (uint8_t)(kind >> 8);
    engram_le_put_u32(buf + 12, key ? ENGRAM_SEAL_F_SEALED : 0u);
    engram_le_put_u64(buf + 16, (uint64_t)n);
    memcpy(buf + 24, s, 32u);
    if (key) {
        uint8_t okm[44];
        engram_seal_derive(key, s, kind, okm);
        engram_aead_seal(okm, okm + 32, buf, ENGRAM_SEAL_HEADER, (const uint8_t *)payload, n,
                         buf + ENGRAM_SEAL_HEADER, buf + ENGRAM_SEAL_HEADER + n);
        memset(buf + ENGRAM_SEAL_HEADER + n + 16u, 0, 16u);
        engram_wipe(okm, sizeof okm);
    } else {
        if (n) memcpy(buf + ENGRAM_SEAL_HEADER, payload, n);
        engram_sha256(buf, ENGRAM_SEAL_HEADER + n, buf + ENGRAM_SEAL_HEADER + n);
    }
    *out = buf;
    *out_n = total;
    return ENGRAM_OK;
}

engram_rc engram_seal_unpack(uint16_t kind, const engram_key *key, const void *file, size_t file_n,
                             uint8_t **payload, size_t *n)
{
    const uint8_t *f = (const uint8_t *)file;
    uint32_t flags;
    uint64_t len;
    uint16_t ver, k;
    uint8_t *pt;
    size_t i;
    if (payload) *payload = NULL;
    if (n) *n = 0;
    if (!payload || !n || (!file && file_n)) return ENGRAM_E_ARG;
    if (file_n < ENGRAM_SEAL_HEADER + ENGRAM_SEAL_TRAILER || memcmp(f, ENGRAM_SEAL_MAGIC, 8u) != 0)
        return ENGRAM_E_FORMAT;
    ver = (uint16_t)(f[8] | (f[9] << 8));
    k = (uint16_t)(f[10] | (f[11] << 8));
    flags = engram_le_get_u32(f + 12);
    len = engram_le_get_u64(f + 16);
    if (ver == 0u) return ENGRAM_E_FORMAT;
    if (ver != ENGRAM_SEAL_VERSION) return ENGRAM_E_VERSION;
    if (k != kind || (flags & ~(uint32_t)ENGRAM_SEAL_F_SEALED) != 0u) return ENGRAM_E_FORMAT;
    if (len != (uint64_t)(file_n - ENGRAM_SEAL_HEADER - ENGRAM_SEAL_TRAILER)) return ENGRAM_E_FORMAT;
    for (i = 56; i < ENGRAM_SEAL_HEADER; i++) if (f[i]) return ENGRAM_E_FORMAT;
    if ((flags & ENGRAM_SEAL_F_SEALED) && !key) return ENGRAM_E_STATE;
    if (!(flags & ENGRAM_SEAL_F_SEALED) && key) return ENGRAM_E_AUTH;      /* no downgrade to plain */
    pt = (uint8_t *)engram_malloc((size_t)len + 1u);
    if (!pt) return ENGRAM_E_MEM;
    if (flags & ENGRAM_SEAL_F_SEALED) {
        uint8_t okm[44];
        engram_rc rc;
        for (i = 0; i < 16u; i++)
            if (f[ENGRAM_SEAL_HEADER + len + 16u + i]) { engram_free(pt); return ENGRAM_E_FORMAT; }
        engram_seal_derive(key, f + 24, k, okm);
        rc = engram_aead_open(okm, okm + 32, f, ENGRAM_SEAL_HEADER, f + ENGRAM_SEAL_HEADER, (size_t)len,
                              f + ENGRAM_SEAL_HEADER + len, pt);
        engram_wipe(okm, sizeof okm);
        if (rc != ENGRAM_OK) { engram_free(pt); return ENGRAM_E_AUTH; }
    } else {
        uint8_t d[32];
        engram_sha256(f, ENGRAM_SEAL_HEADER + (size_t)len, d);
        if (!engram_ct_equal(d, f + ENGRAM_SEAL_HEADER + len, 32u)) { engram_free(pt); return ENGRAM_E_AUTH; }
        if (len) memcpy(pt, f + ENGRAM_SEAL_HEADER, (size_t)len);
    }
    pt[len] = 0u;
    *payload = pt;
    *n = (size_t)len;
    return ENGRAM_OK;
}

engram_rc engram_seal_write(const char *path, uint16_t kind, const engram_key *key, const void *payload, size_t n)
{
    uint8_t *buf = NULL;
    size_t bn = 0;
    engram_rc rc;
    if (!path) return ENGRAM_E_ARG;
    rc = engram_seal_pack(kind, key, NULL, payload, n, &buf, &bn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_file_write_atomic(path, buf, bn);
    engram_free(buf);
    return rc;
}

engram_rc engram_seal_read(const char *path, uint16_t kind, const engram_key *key, uint8_t **payload, size_t *n)
{
    uint8_t *buf = NULL;
    size_t bn = 0;
    engram_rc rc;
    if (payload) *payload = NULL;
    if (n) *n = 0;
    if (!path || !payload || !n) return ENGRAM_E_ARG;
    rc = engram_file_read(path, &buf, &bn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_seal_unpack(kind, key, buf, bn, payload, n);
    engram_free(buf);
    return rc;
}

/* ---- the keyfile ------------------------------------------------------------------------------- */
void engram_keyfile_cfg_default(engram_keyfile_cfg *cfg)
{
    if (!cfg) return;
    cfg->kdf.t_cost = 3u;
    cfg->kdf.m_kib = 64u * 1024u;
    cfg->kdf.lanes = 1u;
}

static int engram_kdf_ok(const engram_argon2_params *p)
{
    return p->t_cost >= 1u && p->t_cost <= ENGRAM_KDF_MAX_PASSES && p->lanes >= 1u &&
           p->lanes <= ENGRAM_KDF_MAX_LANES && p->m_kib >= 8u * p->lanes && p->m_kib <= ENGRAM_KDF_MAX_KIB;
}

static engram_rc engram_stretch(const engram_argon2_params *p, const uint8_t salt[16], const void *pw, size_t plen,
                                engram_key *out, uint8_t verifier[32])
{
    engram_rc rc = engram_argon2id(p, pw, plen, salt, 16u, NULL, 0, "ENGRAM master key", 17u, out->k, sizeof out->k);
    if (rc != ENGRAM_OK) { engram_key_wipe(out); return rc; }
    engram_hmac(out->k, sizeof out->k, ENGRAM_KEYCHECK, sizeof ENGRAM_KEYCHECK - 1u, verifier);
    return ENGRAM_OK;
}

engram_rc engram_keyfile_pack(const engram_keyfile_cfg *cfg, const uint8_t salt[16], const void *password,
                              size_t plen, engram_key *out, uint8_t **file, size_t *file_n)
{
    engram_keyfile_cfg d;
    engram_wbuf w;
    uint8_t verifier[32], s[16], *payload = NULL;
    size_t pn = 0;
    engram_rc rc;
    if (file) *file = NULL;
    if (file_n) *file_n = 0;
    if (out) engram_key_wipe(out);
    if (!out || !file || !file_n || (!password && plen)) return ENGRAM_E_ARG;
    if (!cfg) { engram_keyfile_cfg_default(&d); cfg = &d; }
    if (!engram_kdf_ok(&cfg->kdf)) return ENGRAM_E_ARG;
    if (salt) memcpy(s, salt, 16u);
    else if ((rc = engram_os_random(s, sizeof s)) != ENGRAM_OK) return rc;
    rc = engram_stretch(&cfg->kdf, s, password, plen, out, verifier);
    if (rc != ENGRAM_OK) return rc;
    engram_wbuf_init(&w);
    engram_wbuf_u32(&w, 1u);
    engram_wbuf_u32(&w, cfg->kdf.t_cost);
    engram_wbuf_u32(&w, cfg->kdf.m_kib);
    engram_wbuf_u32(&w, cfg->kdf.lanes);
    engram_wbuf_bytes(&w, s, sizeof s);
    engram_wbuf_bytes(&w, verifier, sizeof verifier);
    rc = engram_wbuf_finish(&w, &payload, &pn);
    if (rc == ENGRAM_OK) rc = engram_seal_pack(ENGRAM_KIND_KEYFILE, NULL, NULL, payload, pn, file, file_n);
    engram_free(payload);
    if (rc != ENGRAM_OK) engram_key_wipe(out);
    return rc;
}

engram_rc engram_keyfile_unpack(const void *file, size_t file_n, const void *password, size_t plen, engram_key *out)
{
    uint8_t *payload = NULL, s[16], ver_file[32], ver_now[32];
    size_t pn = 0;
    engram_rbuf r;
    engram_argon2_params p;
    engram_rc rc;
    if (out) engram_key_wipe(out);
    if (!out || (!password && plen)) return ENGRAM_E_ARG;
    rc = engram_seal_unpack(ENGRAM_KIND_KEYFILE, NULL, file, file_n, &payload, &pn);
    if (rc == ENGRAM_E_AUTH) return ENGRAM_E_FORMAT;     /* a damaged keyfile is not a wrong password */
    if (rc != ENGRAM_OK) return rc;
    engram_rbuf_init(&r, payload, pn);
    if (engram_rbuf_u32(&r) != 1u) engram_rbuf_fail(&r, ENGRAM_E_VERSION);
    p.t_cost = engram_rbuf_u32(&r);
    p.m_kib = engram_rbuf_u32(&r);
    p.lanes = engram_rbuf_u32(&r);
    engram_rbuf_bytes(&r, s, sizeof s);
    engram_rbuf_bytes(&r, ver_file, sizeof ver_file);
    rc = engram_rbuf_end(&r);
    engram_free(payload);
    if (rc != ENGRAM_OK) return rc;
    if (!engram_kdf_ok(&p)) return ENGRAM_E_FORMAT;       /* bounded BEFORE any memory is committed */
    rc = engram_stretch(&p, s, password, plen, out, ver_now);
    if (rc != ENGRAM_OK) return rc;
    if (!engram_ct_equal(ver_file, ver_now, 32u)) { engram_key_wipe(out); return ENGRAM_E_AUTH; }
    return ENGRAM_OK;
}

engram_rc engram_keyfile_create(const char *path, const engram_keyfile_cfg *cfg, const void *password,
                                size_t plen, engram_key *out)
{
    uint8_t *buf = NULL;
    size_t bn = 0;
    engram_rc rc;
    uint64_t size;
    if (out) engram_key_wipe(out);
    if (!path || !out) return ENGRAM_E_ARG;
    /* A cheap refusal before the expensive stretch -- by a check that REPORTS what it cannot tell
     * (engram_file_exists answers "no" to anything it cannot tell, and an overwritten keyfile is every
     * sealed file lost). The guarantee itself is the exclusive commit below. */
    rc = engram_file_size(path, &size);
    if (rc == ENGRAM_OK) return ENGRAM_E_EXISTS;
    if (rc != ENGRAM_E_NOTFOUND) return rc;
    rc = engram_keyfile_pack(cfg, NULL, password, plen, out, &buf, &bn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_file_create_atomic(path, buf, bn);
    engram_free(buf);
    if (rc != ENGRAM_OK) engram_key_wipe(out);
    return rc;
}

engram_rc engram_keyfile_open(const char *path, const void *password, size_t plen, engram_key *out)
{
    uint8_t *buf = NULL;
    size_t bn = 0;
    engram_rc rc;
    if (out) engram_key_wipe(out);
    if (!path || !out) return ENGRAM_E_ARG;
    rc = engram_file_read(path, &buf, &bn);
    if (rc != ENGRAM_OK) return rc;
    rc = engram_keyfile_unpack(buf, bn, password, plen, out);
    engram_free(buf);
    return rc;
}
