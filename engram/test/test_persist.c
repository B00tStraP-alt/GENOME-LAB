/* ==================================================================================================
 * test_persist.c -- P1.5 (part 2): the container, the keyfile, and the store and router on disk.
 * ==================================================================================================
 *   F1  the container against an INDEPENDENT implementation of its format (seal_vectors.h: Python
 *       hashlib/hmac and pyca/cryptography), plain and sealed, five kinds, every length the AEAD and
 *       the hash branch on -- byte for byte, both directions
 *   F2  the keyfile against the same: Argon2id with associated data, the verifier, the payload
 *   F3  refusals: EVERY single-bit change of a plain and of a sealed file refused; every truncation;
 *       an appended byte; the wrong kind; sealed without a key; plain WITH a key (the downgrade); the
 *       wrong key; a sealed file relabelled as another kind; what a plain file can NOT promise
 *   F4  files: write and read; every IO operation of a replacing write failed in turn (old or new, never
 *       torn, no temporary left); every allocation of a write and of a read failed in turn
 *   F5  the keyfile on disk: create, never over an existing one; open with the right and the wrong
 *       password; doctored parameters refused BEFORE memory is committed to them; the default cost
 *   F6  the store: save and load, plain and sealed -- equal fingerprint, counters, episodes and answers;
 *       serialize(load(x)) == x; empty and custom stores; a loaded store keeps working; save is const
 *   F7  the store loader refuses every doctored field (36 cases), every truncation, and survives every
 *       single-byte change of a payload; allocation and IO faults at every point
 *   F8  the router: the same, trained and untrained, with keys added after training and removed; the
 *       loader's refusals; load cost measured
 *   F9  scale: the whole scale corpus saved and loaded, timed; PERSISTPRINT for the gate
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_buf.h"
#include "../src/engram_crypto.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "../src/engram_router.h"
#include "../src/engram_seal.h"
#include "../src/engram_store.h"
#include "seal_vectors.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DIR "engram_persist_scratch"

static int g_quick;

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

static double ms_since(uint64_t t0) { return (double)(engram_now_ns() - t0) / 1e6; }

static uint8_t *dup_bytes(const void *p, size_t n)
{
    uint8_t *d = (uint8_t *)engram_malloc(n ? n : 1u);
    if (d && n) memcpy(d, p, n);
    return d;
}

static int all_zero(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    size_t i;
    for (i = 0; i < n; i++) if (b[i]) return 0;
    return 1;
}

/* the first 8 bytes of a SHA-256, as the 64-bit value the gate compares across platforms */
static uint64_t sha_print(const void *p, size_t n)
{
    uint8_t d[32];
    uint64_t v = 0;
    unsigned i;
    engram_sha256(p, n, d);
    for (i = 0; i < 8u; i++) v = (v << 8) | d[i];
    return v;
}

typedef struct { uint8_t *buf; const char *line[2048]; size_t len[2048]; size_t n; } lines;

static int load_lines(lines *L, const char *name, size_t maxn)
{
    char *path;
    size_t blen = 0, i, s;
    memset(L, 0, sizeof *L);
    path = engram_path_join(data_dir(), name);
    if (!path || engram_file_read(path, &L->buf, &blen) != ENGRAM_OK) { engram_free(path); return 0; }
    engram_free(path);
    for (i = 0; i < blen && L->n < maxn && L->n < 2048u; ) {
        s = i;
        while (i < blen && L->buf[i] != '\n') i++;
        if (i > s) { L->line[L->n] = (const char *)L->buf + s; L->len[L->n] = i - s; L->n++; }
        i++;
    }
    return L->n > 0;
}

/* ==================================================================================================
 * F1, F2: the format against an independent implementation
 * ============================================================================================== */
static void test_vectors(void)
{
    size_t i, bad_pack = 0, bad_unpack = 0;
    ET_SECTION("F1: the container, byte for byte against an independent implementation (plain and sealed)");
    for (i = 0; i < X_SEAL_N; i++) {
        const cv_t *v = &x_seal[i];
        engram_key key;
        uint8_t *f = NULL, *p = NULL;
        size_t fn = 0, pn = 0;
        if (v->y) memcpy(key.k, v->a, 32u);
        if (engram_seal_pack((uint16_t)v->x, v->y ? &key : NULL, v->b, v->c, v->nc, &f, &fn) != ENGRAM_OK ||
            fn != v->nd || memcmp(f, v->d, fn) != 0) bad_pack++;
        if (engram_seal_unpack((uint16_t)v->x, v->y ? &key : NULL, v->d, v->nd, &p, &pn) != ENGRAM_OK ||
            pn != v->nc || (pn && memcmp(p, v->c, pn) != 0) || p[pn] != 0u) bad_unpack++;
        engram_free(f); engram_free(p);
    }
    ET_EQ_U64(bad_pack, 0u);
    ET_EQ_U64(bad_unpack, 0u);
    ET_CHECK(X_SEAL_N >= 90u);
    printf("       %u containers (5 kinds x 9-10 lengths x plain/sealed) identical in both directions\n",
           (unsigned)X_SEAL_N);

    ET_SECTION("F2: the keyfile -- Argon2id with associated data, the verifier, the payload, independently");
    for (i = 0; i < X_KEYFILE_N; i++) {
        const cv_t *v = &x_keyfile[i];
        engram_keyfile_cfg cfg;
        engram_key k1, k2;
        uint8_t *f = NULL, *p = NULL, wrong[512];
        size_t fn = 0, pn = 0;
        cfg.kdf.t_cost = v->x; cfg.kdf.m_kib = v->y; cfg.kdf.lanes = v->z;
        ET_OK(engram_keyfile_pack(&cfg, v->b, v->a, v->na, &k1, &f, &fn));
        ET_CHECK(memcmp(k1.k, v->c, 32u) == 0);
        ET_OK(engram_seal_unpack(ENGRAM_KIND_KEYFILE, NULL, f, fn, &p, &pn));
        ET_CHECK(pn == v->nd && memcmp(p, v->d, pn) == 0);
        ET_OK(engram_keyfile_unpack(f, fn, v->a, v->na, &k2));
        ET_CHECK(memcmp(k2.k, v->c, 32u) == 0);
        /* the wrong password: one byte more, one byte changed */
        if (v->na + 1u > sizeof wrong) { ET_CHECK(v->na + 1u <= sizeof wrong); continue; }
        if (v->na) memcpy(wrong, v->a, v->na);
        wrong[v->na] = 'x';
        ET_RC(engram_keyfile_unpack(f, fn, wrong, v->na + 1u, &k2), ENGRAM_E_AUTH);
        ET_CHECK(all_zero(k2.k, 32u));
        if (v->na) {
            wrong[0] ^= 1u;
            ET_RC(engram_keyfile_unpack(f, fn, wrong, v->na, &k2), ENGRAM_E_AUTH);
            ET_CHECK(all_zero(k2.k, 32u));
        }
        engram_free(f); engram_free(p);
    }
    printf("       %u keyfiles: master key, verifier and payload identical; wrong passwords refused, key wiped\n",
           (unsigned)X_KEYFILE_N);
}

/* ==================================================================================================
 * F3: refusals
 * ============================================================================================== */
static void flip_sweep(const uint8_t *file, size_t n, uint16_t kind, const engram_key *key, const char *what)
{
    uint8_t *m = dup_bytes(file, n), *p = NULL;
    size_t bit, pn = 0, accepted = 0, leaked = 0, tally[16];
    engram_rc rc;
    memset(tally, 0, sizeof tally);
    if (!m) { ET_CHECK(m != NULL); return; }
    /* R6: the untouched file is accepted, so a refusal below is the change's doing */
    ET_OK(engram_seal_unpack(kind, key, m, n, &p, &pn));
    engram_free(p);
    for (bit = 0; bit < n * 8u; bit++) {
        m[bit / 8u] ^= (uint8_t)(1u << (bit % 8u));
        p = (uint8_t *)1;
        rc = engram_seal_unpack(kind, key, m, n, &p, &pn);
        if (rc == ENGRAM_OK) { accepted++; engram_free(p); }
        else if (p != NULL || pn != 0u) leaked++;
        tally[(unsigned)(-rc) & 15u]++;
        m[bit / 8u] ^= (uint8_t)(1u << (bit % 8u));
    }
    ET_EQ_U64(accepted, 0u);
    ET_EQ_U64(leaked, 0u);
    printf("       %s: %u flips refused -- FORMAT %u, VERSION %u, STATE %u, AUTH %u\n", what, (unsigned)(n * 8u),
           (unsigned)tally[4], (unsigned)tally[5], (unsigned)tally[12], (unsigned)tally[9]);
    engram_free(m);
}

static void test_refusals(void)
{
    engram_key key, other;
    uint8_t salt[32], payload[40], *plain = NULL, *sealed = NULL, *p = NULL, *f2 = NULL, *big = NULL;
    size_t pln = 0, sln = 0, pn = 0, f2n = 0, cut, bad = 0, i;
    engram_rc rc;
    for (i = 0; i < 32u; i++) { key.k[i] = (uint8_t)(i * 7u + 3u); other.k[i] = key.k[i]; salt[i] = (uint8_t)(0xA0u + i); }
    other.k[31] ^= 0x01u;
    for (i = 0; i < sizeof payload; i++) payload[i] = (uint8_t)("a small memory, forty bytes long exactly"[i]);
    ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, NULL, salt, payload, sizeof payload, &plain, &pln));
    ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, &key, salt, payload, sizeof payload, &sealed, &sln));
    if (!plain || !sealed) goto out;

    ET_SECTION("F3: every single-bit change of a plain and of a sealed file is refused, nothing handed back");
    flip_sweep(plain, pln, ENGRAM_KIND_STORE, NULL, "plain ");
    flip_sweep(sealed, sln, ENGRAM_KIND_STORE, &key, "sealed");

    ET_SECTION("F3: every truncation, and an appended byte, refused");
    for (cut = 0; cut < sln; cut++) {
        if (engram_seal_unpack(ENGRAM_KIND_STORE, &key, sealed, cut, &p, &pn) == ENGRAM_OK) { bad++; engram_free(p); }
        if (engram_seal_unpack(ENGRAM_KIND_STORE, NULL, plain, cut, &p, &pn) == ENGRAM_OK) { bad++; engram_free(p); }
    }
    ET_EQ_U64(bad, 0u);
    big = (uint8_t *)engram_malloc(sln + 1u);
    if (big) {
        memcpy(big, sealed, sln); big[sln] = 0u;
        ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, big, sln + 1u, &p, &pn), ENGRAM_E_FORMAT);
        memcpy(big, plain, pln); big[pln] = 0u;
        ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, big, pln + 1u, &p, &pn), ENGRAM_E_FORMAT);
    }

    ET_SECTION("F3: the wrong kind; sealed without a key; plain WITH a key (no downgrade); the wrong key");
    ET_RC(engram_seal_unpack(ENGRAM_KIND_ROUTER, &key, sealed, sln, &p, &pn), ENGRAM_E_FORMAT);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_ROUTER, NULL, plain, pln, &p, &pn), ENGRAM_E_FORMAT);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, sealed, sln, &p, &pn), ENGRAM_E_STATE);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, plain, pln, &p, &pn), ENGRAM_E_AUTH);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &other, sealed, sln, &p, &pn), ENGRAM_E_AUTH);
    ET_CHECK(p == NULL && pn == 0u);

    ET_SECTION("F3: a sealed file relabelled as another kind fails its tag -- the kind is bound in");
    {
        uint8_t *m = dup_bytes(sealed, sln);
        if (m) {
            m[10] = (uint8_t)ENGRAM_KIND_ROUTER; m[11] = 0u;
            ET_RC(engram_seal_unpack(ENGRAM_KIND_ROUTER, &key, m, sln, &p, &pn), ENGRAM_E_AUTH);
            engram_free(m);
        }
    }

    ET_SECTION("F3: what PLAIN does not promise -- a deliberate edit with the hash recomputed is accepted");
    {   /* the reason a caller holding a key refuses plain files: this is exactly what it would accept */
        uint8_t *m = dup_bytes(plain, pln);
        if (m) {
            m[64] ^= 0x20u;
            engram_sha256(m, 64u + sizeof payload, m + 64u + sizeof payload);
            ET_OK(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, m, pln, &p, &pn));
            ET_CHECK(p && p[0] == (uint8_t)(payload[0] ^ 0x20u));
            engram_free(p); p = NULL;
            ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, pln, &p, &pn), ENGRAM_E_AUTH);
            engram_free(m);
        }
    }

    ET_SECTION("F3: a fresh salt per write -- the same payload and key never give the same file");
    ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, &key, NULL, payload, sizeof payload, &f2, &f2n));
    if (f2) {
        uint8_t *f3 = NULL;
        size_t f3n = 0;
        ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, &key, NULL, payload, sizeof payload, &f3, &f3n));
        ET_CHECK(f3 && f3n == f2n && memcmp(f2 + 24, f3 + 24, 32u) != 0 && memcmp(f2 + 64, f3 + 64, sizeof payload) != 0);
        ET_OK(engram_seal_unpack(ENGRAM_KIND_STORE, &key, f3, f3n, &p, &pn));
        ET_CHECK(p && pn == sizeof payload && memcmp(p, payload, pn) == 0);
        engram_free(p); p = NULL;
        engram_free(f3);
        /* and the ciphertext is not the plaintext */
        ET_CHECK(memcmp(f2 + 64, payload, sizeof payload) != 0);
        engram_free(f2);
        f2 = NULL;
    }

    ET_SECTION("F3: version, flags, reserved bytes and the sealed trailer's padding are all checked");
    {
        uint8_t *m = dup_bytes(sealed, sln);
        if (m) {
            m[8] = 2u;                                   rc = engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn);
            ET_RC(rc, ENGRAM_E_VERSION);                 m[8] = 0u;
            ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn), ENGRAM_E_FORMAT);
            m[8] = 1u; m[13] = 0x01u;                     /* an unknown flag bit */
            ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn), ENGRAM_E_FORMAT);
            m[13] = 0u; m[60] = 1u;                       /* reserved */
            ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn), ENGRAM_E_FORMAT);
            m[60] = 0u; m[sln - 1u] = 1u;                 /* the trailer's zero padding */
            ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn), ENGRAM_E_FORMAT);
            m[sln - 1u] = 0u;
            ET_OK(engram_seal_unpack(ENGRAM_KIND_STORE, &key, m, sln, &p, &pn));   /* restored: accepted */
            engram_free(p); p = NULL;
            engram_free(m);
        }
    }

    ET_SECTION("F3: argument refusals");
    ET_RC(engram_seal_pack(0u, NULL, NULL, payload, 1u, &f2, &f2n), ENGRAM_E_ARG);
    ET_CHECK(f2 == NULL && f2n == 0u);
    ET_RC(engram_seal_pack(ENGRAM_KIND_STORE, NULL, NULL, NULL, 1u, &f2, &f2n), ENGRAM_E_ARG);
    ET_RC(engram_seal_pack(ENGRAM_KIND_STORE, NULL, NULL, payload, 1u, NULL, &f2n), ENGRAM_E_ARG);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, NULL, 5u, &p, &pn), ENGRAM_E_ARG);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, plain, pln, NULL, &pn), ENGRAM_E_ARG);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, "ENGRAM", 6u, &p, &pn), ENGRAM_E_FORMAT);
    ET_RC(engram_seal_unpack(ENGRAM_KIND_STORE, NULL, NULL, 0u, &p, &pn), ENGRAM_E_FORMAT);
out:
    engram_free(plain); engram_free(sealed); engram_free(f2); engram_free(big);
}

/* ==================================================================================================
 * F4: files, and faults
 * ============================================================================================== */
static unsigned orphans_cb_n;
static int orphans_cb(const char *name, void *user)
{
    (void)user;
    if (engram_tmp_name_is_orphan(name)) orphans_cb_n++;
    return 0;
}
static unsigned count_orphans(void)
{
    orphans_cb_n = 0;
    if (engram_dir_list(DIR, orphans_cb, NULL) != ENGRAM_OK) return 9999u;
    return orphans_cb_n;
}

static void test_files(void)
{
    const char *path = DIR "/sealed.bin";
    const char *oldp = "the old contents, which must survive any failed write";
    const char *newp = "the new contents -- they arrive whole, or not at all, and never torn apart";
    engram_key key;
    uint8_t *p = NULL;
    size_t pn = 0, i, lo = strlen(oldp), ln = strlen(newp);
    uint64_t ops, k, calls;
    unsigned torn = 0, silent = 0, old_lost = 0, is_new_n = 0;
    for (i = 0; i < 32u; i++) key.k[i] = (uint8_t)(0x5Au ^ i);

    ET_SECTION("F4: write and read back, plain and sealed; an absent file is NOTFOUND");
    ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, NULL, oldp, lo));
    ET_OK(engram_seal_read(path, ENGRAM_KIND_TEST, NULL, &p, &pn));
    ET_CHECK(p && pn == lo && memcmp(p, oldp, lo) == 0);
    engram_free(p); p = NULL;
    ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, &key, oldp, lo));
    ET_OK(engram_seal_read(path, ENGRAM_KIND_TEST, &key, &p, &pn));
    ET_CHECK(p && pn == lo && memcmp(p, oldp, lo) == 0);
    engram_free(p); p = NULL;
    ET_RC(engram_seal_read(DIR "/absent.bin", ENGRAM_KIND_TEST, &key, &p, &pn), ENGRAM_E_NOTFOUND);
    ET_RC(engram_seal_write(NULL, ENGRAM_KIND_TEST, &key, oldp, lo), ENGRAM_E_ARG);
    ET_RC(engram_seal_read(NULL, ENGRAM_KIND_TEST, &key, &p, &pn), ENGRAM_E_ARG);

    ET_SECTION("F4: every IO operation of a replacing sealed write failed in turn -- old or new, never torn");
    engram_io_reset();
    ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, &key, newp, ln));
    ops = engram_io_ops();
    for (k = 1; k <= ops; k++) {
        engram_rc rc;
        int is_old, is_new;
        ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, &key, oldp, lo));
        engram_io_reset();
        engram_io_fail_at(k);
        rc = engram_seal_write(path, ENGRAM_KIND_TEST, &key, newp, ln);
        engram_io_fail_at(0u);
        if (engram_seal_read(path, ENGRAM_KIND_TEST, &key, &p, &pn) != ENGRAM_OK) { torn++; continue; }
        is_old = pn == lo && memcmp(p, oldp, lo) == 0;
        is_new = pn == ln && memcmp(p, newp, ln) == 0;
        engram_free(p); p = NULL;
        if (!is_old && !is_new) torn++;
        if (rc == ENGRAM_OK) silent++;
        if (is_new) is_new_n++;
        if (k < ops && !is_old) old_lost++;
    }
    ET_EQ_U64(torn, 0u);
    ET_EQ_U64(silent, 0u);
    ET_EQ_U64(old_lost, 0u);
    ET_EQ_U64(count_orphans(), 0u);
    printf("       %llu IO operations per write, each failed in turn: 0 torn, 0 unreported, %u after the commit point\n",
           (unsigned long long)ops, is_new_n);

    ET_SECTION("F4: every allocation of a write and of a read failed in turn -- E_MEM, the file unchanged");
    engram_alloc_reset_run();
    ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, &key, oldp, lo));
    calls = engram_alloc_calls();
    for (k = 1; k <= calls + 1u; k++) {
        engram_rc rc;
        engram_alloc_reset_run();
        engram_alloc_fail_at(k);
        rc = engram_seal_write(path, ENGRAM_KIND_TEST, &key, newp, ln);
        engram_alloc_fail_at(0u);
        ET_CHECK(rc == (k <= calls ? ENGRAM_E_MEM : ENGRAM_OK));
        if (k <= calls) {
            ET_OK(engram_seal_read(path, ENGRAM_KIND_TEST, &key, &p, &pn));
            ET_CHECK(p && pn == lo && memcmp(p, oldp, lo) == 0);
            engram_free(p); p = NULL;
        }
        ET_OK(engram_seal_write(path, ENGRAM_KIND_TEST, &key, oldp, lo));
    }
    engram_alloc_reset_run();
    ET_OK(engram_seal_read(path, ENGRAM_KIND_TEST, &key, &p, &pn));
    engram_free(p); p = NULL;
    calls = engram_alloc_calls();
    for (k = 1; k <= calls; k++) {
        engram_alloc_reset_run();
        engram_alloc_fail_at(k);
        ET_RC(engram_seal_read(path, ENGRAM_KIND_TEST, &key, &p, &pn), ENGRAM_E_MEM);
        engram_alloc_fail_at(0u);
        ET_CHECK(p == NULL);
    }
    ET_EQ_U64(count_orphans(), 0u);
    ET_OK(engram_file_remove(path));
}

/* ==================================================================================================
 * F5: the keyfile on disk
 * ============================================================================================== */
static void keyfile_doctored(const uint32_t v[4], const char *pw, engram_rc want, const char *what)
{
    engram_wbuf w;
    uint8_t *pl = NULL, *f = NULL, s[16], ver[32];
    size_t pln = 0, fn = 0;
    engram_key k;
    engram_alloc_stats a0, a1;
    memset(s, 7, sizeof s); memset(ver, 9, sizeof ver);
    engram_wbuf_init(&w);
    engram_wbuf_u32(&w, v[0]); engram_wbuf_u32(&w, v[1]); engram_wbuf_u32(&w, v[2]); engram_wbuf_u32(&w, v[3]);
    engram_wbuf_bytes(&w, s, sizeof s);
    engram_wbuf_bytes(&w, ver, sizeof ver);
    if (engram_wbuf_finish(&w, &pl, &pln) != ENGRAM_OK) { ET_CHECK(0); return; }
    ET_OK(engram_seal_pack(ENGRAM_KIND_KEYFILE, NULL, NULL, pl, pln, &f, &fn));
    engram_alloc_reset_run();
    engram_alloc_stats_get(&a0);
    ET_CHECKF(engram_keyfile_unpack(f, fn, pw, strlen(pw), &k) == want, "%s", what);
    engram_alloc_stats_get(&a1);
    /* refused before any memory was committed to the doctored parameters: nothing near their size */
    if (want != ENGRAM_E_AUTH)
        ET_CHECKF(a1.peak_bytes - a0.live_bytes < 4096u + 2u * fn, "%s: peak %llu B above the start", what,
                  (unsigned long long)(a1.peak_bytes - a0.live_bytes));
    ET_CHECK(all_zero(k.k, 32u));
    engram_free(pl); engram_free(f);
}

static void test_keyfile(void)
{
    const char *path = DIR "/key.engram";
    const char *pw = "correct horse battery staple";
    engram_keyfile_cfg cfg;
    engram_key k1, k2;
    uint8_t *before = NULL, *after = NULL;
    size_t bn = 0, an = 0;
    uint64_t t0;

    ET_SECTION("F5: create, then open with the right password -- the same master key");
    (void)engram_file_remove(path);
    cfg.kdf.t_cost = 1u; cfg.kdf.m_kib = 64u; cfg.kdf.lanes = 1u;
    ET_OK(engram_keyfile_create(path, &cfg, pw, strlen(pw), &k1));
    ET_OK(engram_keyfile_open(path, pw, strlen(pw), &k2));
    ET_CHECK(memcmp(k1.k, k2.k, 32u) == 0 && !all_zero(k1.k, 32u));

    ET_SECTION("F5: never created over an existing keyfile -- E_EXISTS, the file byte-identical");
    ET_OK(engram_file_read(path, &before, &bn));
    ET_RC(engram_keyfile_create(path, &cfg, "another password", 16u, &k2), ENGRAM_E_EXISTS);
    ET_OK(engram_file_read(path, &after, &an));
    ET_CHECK(before && after && bn == an && memcmp(before, after, bn) == 0);
    engram_free(before); engram_free(after);

    ET_SECTION("F5: the wrong password is E_AUTH -- reported as a wrong password, the key wiped");
    memset(k2.k, 0xEE, sizeof k2.k);
    ET_RC(engram_keyfile_open(path, "correct horse battery stapler", 29u, &k2), ENGRAM_E_AUTH);
    ET_CHECK(all_zero(k2.k, 32u));
    ET_RC(engram_keyfile_open(path, "", 0u, &k2), ENGRAM_E_AUTH);
    ET_RC(engram_keyfile_open(DIR "/no-such-keyfile", pw, strlen(pw), &k2), ENGRAM_E_NOTFOUND);
    ET_RC(engram_keyfile_open(path, NULL, 3u, &k2), ENGRAM_E_ARG);
    ET_RC(engram_keyfile_create(NULL, &cfg, pw, 3u, &k2), ENGRAM_E_ARG);

    ET_SECTION("F5: a sealed store cannot be opened with the key of another password");
    {
        uint8_t *p = NULL;
        size_t pn = 0;
        ET_OK(engram_seal_write(DIR "/k.bin", ENGRAM_KIND_TEST, &k1, "under the right key", 19u));
        ET_OK(engram_keyfile_pack(&cfg, NULL, "a different password", 20u, &k2, &before, &bn));
        engram_free(before); before = NULL;
        ET_RC(engram_seal_read(DIR "/k.bin", ENGRAM_KIND_TEST, &k2, &p, &pn), ENGRAM_E_AUTH);
        ET_OK(engram_seal_read(DIR "/k.bin", ENGRAM_KIND_TEST, &k1, &p, &pn));
        engram_free(p);
        ET_OK(engram_file_remove(DIR "/k.bin"));
    }

    ET_SECTION("F5: doctored parameters are refused BEFORE memory is committed to them");
    {
        static const uint32_t cases[][4] = {
            { 1u, 0u, 64u, 1u },                               /* no passes                    */
            { 1u, 65u, 64u, 1u },                              /* more passes than the bound   */
            { 1u, 1u, 64u, 0u },                               /* no lanes                     */
            { 1u, 1u, 256u, 17u },                             /* more lanes than the bound    */
            { 1u, 1u, 7u, 1u },                                /* memory under 8 x lanes       */
            { 1u, 1u, 31u, 4u },
            { 1u, 1u, 4u * 1024u * 1024u + 1u, 1u },           /* 4 GiB + 1 KiB                */
            { 1u, 1u, 0xFFFFFFFFu, 1u },
            { 1u, 0xFFFFFFFFu, 64u, 1u },
        };
        static const char *names[] = { "t 0", "t 65", "lanes 0", "lanes 17", "m 7", "m 31 x 4 lanes",
                                       "m 4 GiB + 1", "m 2^32 - 1", "t 2^32 - 1" };
        unsigned i;
        uint32_t v2[4] = { 2u, 1u, 64u, 1u }, ok[4] = { 1u, 1u, 64u, 1u };
        for (i = 0; i < sizeof cases / sizeof cases[0]; i++) keyfile_doctored(cases[i], pw, ENGRAM_E_FORMAT, names[i]);
        keyfile_doctored(v2, pw, ENGRAM_E_VERSION, "payload version 2");
        keyfile_doctored(ok, pw, ENGRAM_E_AUTH, "valid parameters, a verifier of the wrong password");
    }
    {
        uint8_t *f = NULL, *p = NULL, *g = NULL;
        size_t fn = 0, pn = 0, gn = 0;
        ET_OK(engram_file_read(path, &f, &fn));
        ET_OK(engram_seal_unpack(ENGRAM_KIND_KEYFILE, NULL, f, fn, &p, &pn));
        if (p) {
            ET_OK(engram_seal_pack(ENGRAM_KIND_KEYFILE, NULL, NULL, p, pn - 1u, &g, &gn));   /* short */
            ET_RC(engram_keyfile_unpack(g, gn, pw, strlen(pw), &k2), ENGRAM_E_FORMAT);
            engram_free(g); g = NULL;
            ET_OK(engram_seal_pack(ENGRAM_KIND_KEYFILE, &k1, NULL, p, pn, &g, &gn));       /* sealed */
            ET_RC(engram_keyfile_unpack(g, gn, pw, strlen(pw), &k2), ENGRAM_E_STATE);
            engram_free(g); g = NULL;
            ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, NULL, NULL, p, pn, &g, &gn));        /* another kind */
            ET_RC(engram_keyfile_unpack(g, gn, pw, strlen(pw), &k2), ENGRAM_E_FORMAT);
            engram_free(g); g = NULL;
        }
        f[fn - 1u] ^= 1u;                                           /* damaged: not a wrong password */
        ET_RC(engram_keyfile_unpack(f, fn, pw, strlen(pw), &k2), ENGRAM_E_FORMAT);
        engram_free(f); engram_free(p);
    }

    ET_SECTION("F5: allocation failure in create and open -- E_MEM, the key wiped, no file left behind");
    {
        uint64_t calls, k;
        engram_rc rc;
        (void)engram_file_remove(path);
        engram_alloc_reset_run();
        ET_OK(engram_keyfile_create(path, &cfg, pw, strlen(pw), &k1));
        calls = engram_alloc_calls();
        ET_OK(engram_file_remove(path));
        for (k = 1; k <= calls; k++) {
            engram_alloc_reset_run();
            engram_alloc_fail_at(k);
            memset(k2.k, 0xEE, 32u);
            rc = engram_keyfile_create(path, &cfg, pw, strlen(pw), &k2);
            engram_alloc_fail_at(0u);
            ET_RC(rc, ENGRAM_E_MEM);
            ET_CHECK(all_zero(k2.k, 32u) && !engram_file_exists(path));
        }
        ET_OK(engram_keyfile_create(path, &cfg, pw, strlen(pw), &k1));
        engram_alloc_reset_run();
        ET_OK(engram_keyfile_open(path, pw, strlen(pw), &k2));
        calls = engram_alloc_calls();
        for (k = 1; k <= calls; k++) {
            engram_alloc_reset_run();
            engram_alloc_fail_at(k);
            memset(k2.k, 0xEE, 32u);
            rc = engram_keyfile_open(path, pw, strlen(pw), &k2);
            engram_alloc_fail_at(0u);
            ET_RC(rc, ENGRAM_E_MEM);
            ET_CHECK(all_zero(k2.k, 32u));
        }
    }

    ET_SECTION("F5: the DEFAULT cost (3 passes over 64 MiB) -- works, and what it costs here");
    engram_keyfile_cfg_default(&cfg);
    ET_CHECK(cfg.kdf.t_cost == 3u && cfg.kdf.m_kib == 65536u && cfg.kdf.lanes == 1u);
    ET_OK(engram_file_remove(path));
    t0 = engram_now_ns();
    ET_OK(engram_keyfile_create(path, NULL, pw, strlen(pw), &k1));
    printf("       default stretch: %.0f ms to create (one Argon2id)\n", ms_since(t0));
    t0 = engram_now_ns();
    ET_OK(engram_keyfile_open(path, pw, strlen(pw), &k2));
    printf("       default stretch: %.0f ms to open\n", ms_since(t0));
    ET_CHECK(memcmp(k1.k, k2.k, 32u) == 0);
    ET_OK(engram_file_remove(path));
    engram_key_wipe(&k1); engram_key_wipe(&k2);
    ET_CHECK(all_zero(k1.k, 32u));
}

/* ==================================================================================================
 * F6, F7: the store
 * ============================================================================================== */
static lines g_en, g_de, g_zh;

/* A store with a history: adds in three languages, consolidations, deletions, recalls that move the
 * counters, and -- when the caps are small -- evictions and compactions. */
static engram_store *build_store(const engram_store_cfg *cfg, size_t n_en)
{
    engram_store *s = NULL;
    engram_hit h[4];
    uint64_t first, id;
    size_t na, nh, i;
    if (engram_store_open(&s, cfg) != ENGRAM_OK) return NULL;
    for (i = 0; i < n_en && i < g_en.n; i++) {
        (void)engram_store_add(s, g_en.line[i], g_en.len[i], 1000u + i, (uint32_t)(i % 3u), &first, &na);
        if (i % 4u == 1u) (void)engram_store_consolidated(s, first);
    }
    for (i = 0; i < 12u && i < g_de.n; i++) (void)engram_store_add(s, g_de.line[i], g_de.len[i], 5000u + i, 7u, &first, &na);
    for (i = 0; i < 12u && i < g_zh.n; i++) (void)engram_store_add(s, g_zh.line[i], g_zh.len[i], 6000u + i, 8u, &first, &na);
    for (id = 3u; id < 60u; id += 7u) (void)engram_store_delete(s, id);
    for (i = 0; i < 20u && i < g_en.n; i++) {
        size_t off = g_en.len[i] / 3u, len = g_en.len[i] - off < 40u ? g_en.len[i] - off : 40u;
        (void)engram_store_recall(s, g_en.line[i] + off, len, h, 4u, &nh);
    }
    return s;
}

static void stats_equal(const engram_store *a, const engram_store *b, const char *what)
{
    engram_store_stats x, y;
    engram_store_stats_get(a, &x);
    engram_store_stats_get(b, &y);
    ET_CHECKF(x.live == y.live && x.text_bytes == y.text_bytes && x.consolidated == y.consolidated &&
              x.next_id == y.next_id && x.adds == y.adds && x.deletes == y.deletes && x.evictions == y.evictions &&
              x.refusals == y.refusals && x.compactions == y.compactions && y.tombstones == 0u,
              "%s: live %zu/%zu bytes %zu/%zu cons %zu/%zu next %llu/%llu evict %llu/%llu", what, x.live, y.live,
              x.text_bytes, y.text_bytes, x.consolidated, y.consolidated, (unsigned long long)x.next_id,
              (unsigned long long)y.next_id, (unsigned long long)x.evictions, (unsigned long long)y.evictions);
}

/* The same answers to the same cues, field for field, bit for bit -- then the same fingerprint, since
 * recall moved the same counters in both. */
static void answers_equal(engram_store *a, engram_store *b, const char *what)
{
    size_t i, bad = 0, asked = 0;
    for (i = 0; i < g_en.n && i < 60u; i++) {
        engram_hit ha[5], hb[5];
        size_t na = 0, nb = 0, off = (g_en.len[i] * (i % 5u)) / 7u, len = g_en.len[i] - off;
        engram_rc ra, rb;
        if (len > 16u + (i % 50u)) len = 16u + (i % 50u);
        ra = engram_store_recall(a, g_en.line[i] + off, len, ha, 5u, &na);
        rb = engram_store_recall(b, g_en.line[i] + off, len, hb, 5u, &nb);
        asked++;
        if (ra != rb || na != nb || (na && memcmp(ha, hb, na * sizeof ha[0]) != 0)) bad++;
    }
    ET_CHECKF(bad == 0u, "%s: %zu of %zu answers differ", what, bad, asked);
    ET_EQ_U64(engram_store_fingerprint(a), engram_store_fingerprint(b));
}

static void episodes_equal(const engram_store *a, const engram_store *b, const char *what)
{
    engram_store_stats st;
    uint64_t id;
    size_t bad = 0, seen = 0;
    engram_store_stats_get(a, &st);
    for (id = 1u; id < st.next_id; id++) {
        engram_episode x, y;
        engram_rc ra = engram_store_get(a, id, &x), rb = engram_store_get(b, id, &y);
        if (ra != rb) { bad++; continue; }
        if (ra != ENGRAM_OK) continue;
        seen++;
        if (x.id != y.id || x.time_ms != y.time_ms || x.source != y.source || x.flags != y.flags ||
            x.recalls != y.recalls || x.len != y.len || memcmp(x.text, y.text, x.len) != 0) bad++;
    }
    ET_CHECKF(bad == 0u && seen == st.live, "%s: %zu differ, %zu of %zu seen", what, bad, seen, st.live);
}

static void roundtrip_store(engram_store *s, const engram_key *key, const char *what)
{
    const char *path = DIR "/store.engram";
    engram_store *l = NULL;
    uint8_t *x = NULL, *y = NULL;
    size_t xn = 0, yn = 0;
    uint64_t fp = engram_store_fingerprint(s);
    ET_OK(engram_store_save(s, path, key));
    ET_EQ_U64(engram_store_fingerprint(s), fp);                      /* save touched nothing */
    ET_OK(engram_store_load(&l, path, key));
    if (!l) return;
    ET_EQ_U64(engram_store_fingerprint(l), fp);
    stats_equal(s, l, what);
    episodes_equal(s, l, what);
    ET_OK(engram_store_serialize(s, &x, &xn));
    ET_OK(engram_store_serialize(l, &y, &yn));
    ET_CHECKF(x && y && xn == yn && memcmp(x, y, xn) == 0, "%s: serialize(load(x)) != x", what);
    engram_free(x); engram_free(y);
    answers_equal(s, l, what);
    {   /* and the loaded store keeps working: the same add gives the same ids and the same state */
        uint64_t f1 = 0, f2 = 0;
        size_t n1 = 0, n2 = 0;
        const char *t = "A memory added after the load: the river froze early that winter.";
        ET_OK(engram_store_add(s, t, strlen(t), 99999u, 5u, &f1, &n1));
        ET_OK(engram_store_add(l, t, strlen(t), 99999u, 5u, &f2, &n2));
        ET_CHECK(f1 == f2 && n1 == n2);
        ET_EQ_U64(engram_store_fingerprint(l), engram_store_fingerprint(s));
    }
    engram_store_close(l);
}

static void test_store_roundtrip(void)
{
    engram_store_cfg cfg;
    engram_store *s;
    engram_key key;
    unsigned i;
    for (i = 0; i < 32u; i++) key.k[i] = (uint8_t)(i * 13u + 1u);

    ET_SECTION("F6: a store with a history -- saved and loaded, plain and sealed: fingerprint, counters, episodes, answers");
    engram_store_cfg_default(&cfg);
    s = build_store(&cfg, 200u);
    if (s) {
        engram_store_stats st;
        engram_store_stats_get(s, &st);
        ET_CHECK(st.live > 150u && st.deletes > 0u && st.consolidated > 0u);
        roundtrip_store(s, NULL, "plain");
        roundtrip_store(s, &key, "sealed");
        engram_store_close(s);
    }

    ET_SECTION("F6: a store that evicted and compacted -- its lifetime counters survive the round trip");
    engram_store_cfg_default(&cfg);
    cfg.max_episodes = 60u;
    cfg.max_text_bytes = 12000u;
    s = build_store(&cfg, 150u);
    if (s) {
        engram_store_stats st;
        size_t k;
        uint64_t first;
        size_t na;
        for (k = 0; k < 150u && k < g_en.n; k++) {       /* consolidate everything, add more: evictions */
            engram_store_stats_get(s, &st);
            (void)engram_store_consolidated(s, st.next_id - 1u - (k % 30u));
        }
        for (k = 150u; k < 260u && k < g_en.n; k++) {
            uint64_t id;
            engram_store_stats_get(s, &st);
            for (id = 1u; id < st.next_id; id++) (void)engram_store_consolidated(s, id);
            (void)engram_store_add(s, g_en.line[k], g_en.len[k], 9000u + k, 1u, &first, &na);
        }
        engram_store_stats_get(s, &st);
        /* every lifetime counter non-zero, so each one's survival is actually tested */
        ET_CHECKF(st.evictions > 0u && st.compactions > 0u && st.refusals > 0u && st.deletes > 0u && st.adds > 0u,
                  "evictions %llu compactions %llu refusals %llu deletes %llu", (unsigned long long)st.evictions,
                  (unsigned long long)st.compactions, (unsigned long long)st.refusals, (unsigned long long)st.deletes);
        roundtrip_store(s, &key, "evicting");
        engram_store_close(s);
    }

    ET_SECTION("F6: a custom configuration is part of the file (rank, C, chunking, fold, weights, UTF-8 policy)");
    engram_store_cfg_default(&cfg);
    cfg.rank = ENGRAM_RANK_BAG;
    cfg.recall_c = 17u;
    cfg.chunk_cap = 200u;
    cfg.chunk_overlap = 20u;
    cfg.enc.fold = ENGRAM_FOLD_CASE;
    cfg.enc.w_word1 = 0.5f;
    cfg.enc.utf8 = ENGRAM_UTF8_STRICT;
    cfg.enc.seed ^= 0x1234u;
    s = build_store(&cfg, 80u);
    if (s) {
        roundtrip_store(s, NULL, "custom");
        engram_store_close(s);
    }

    ET_SECTION("F6: the empty store");
    s = NULL;
    ET_OK(engram_store_open(&s, NULL));
    if (s) {
        roundtrip_store(s, &key, "empty");
        engram_store_close(s);
    }

    ET_SECTION("F6: refusals -- the wrong key, no key, the downgrade, the wrong kind, bad arguments");
    ET_OK(engram_store_open(&s, NULL));
    if (s) {
        engram_store *l = (engram_store *)1;
        engram_key other = key;
        engram_router *r = (engram_router *)1;
        other.k[0] ^= 1u;
        ET_OK(engram_store_save(s, DIR "/s.engram", &key));
        ET_RC(engram_store_load(&l, DIR "/s.engram", &other), ENGRAM_E_AUTH);
        ET_CHECK(l == NULL);
        ET_RC(engram_store_load(&l, DIR "/s.engram", NULL), ENGRAM_E_STATE);
        ET_RC(engram_router_load(&r, DIR "/s.engram", &key), ENGRAM_E_FORMAT);
        ET_CHECK(r == NULL);
        ET_OK(engram_store_save(s, DIR "/s.engram", NULL));
        ET_RC(engram_store_load(&l, DIR "/s.engram", &key), ENGRAM_E_AUTH);
        ET_RC(engram_store_save(NULL, DIR "/s.engram", NULL), ENGRAM_E_ARG);
        ET_RC(engram_store_save(s, NULL, NULL), ENGRAM_E_ARG);
        ET_RC(engram_store_load(NULL, DIR "/s.engram", NULL), ENGRAM_E_ARG);
        ET_RC(engram_store_load(&l, DIR "/absent.engram", NULL), ENGRAM_E_NOTFOUND);
        ET_RC(engram_store_deserialize(&l, NULL, 4u), ENGRAM_E_ARG);
        ET_RC(engram_store_serialize(NULL, NULL, NULL), ENGRAM_E_ARG);
        ET_OK(engram_file_remove(DIR "/s.engram"));
        engram_store_close(s);
    }
}

/* ---- F7: the loader against doctored payloads ----------------------------------------------------
 * The payload's layout (engram_store.c): the fixed part is 140 bytes, then 32 bytes + text per episode. */
enum {
    O_VER = 0, O_MAXEP = 4, O_MAXTB = 12, O_CAP = 20, O_OVL = 24, O_C = 28, O_RANK = 32, O_W = 36,
    O_PAIRS = 60, O_FOLD = 64, O_UTF8 = 68, O_SEED = 72, O_TF = 80, O_NEXT = 84, O_LIVE = 132, O_REC = 140
};
#define R_ID 0
#define R_FLAGS 20
#define R_LEN 28
#define R_TEXT 32

typedef struct { unsigned off, width; uint64_t v; engram_rc want; const char *what; } patch;

static engram_rc try_payload(const uint8_t *p, size_t n)
{
    engram_store *s = (engram_store *)1;
    engram_rc rc = engram_store_deserialize(&s, p, n);
    if (rc == ENGRAM_OK) engram_store_close(s);
    else if (s != NULL) return ENGRAM_E_INTERNAL;           /* a refusal must hand back nothing */
    return rc;
}

static void put(uint8_t *p, unsigned width, uint64_t v)
{
    if (width == 4u) engram_le_put_u32(p, (uint32_t)v); else engram_le_put_u64(p, v);
}

static void test_store_doctored(void)
{
    static const char *e0 = "the first episode of the doctored store";
    static const char *e1 = "a second memory, about rivers and boats";
    engram_store *s = NULL;
    uint8_t *x = NULL, *m = NULL;
    size_t xn = 0, l0 = strlen(e0), l1 = strlen(e1), r1, i, bad;
    uint64_t first;
    size_t na;
    ET_SECTION("F7: the loader refuses every doctored field -- and hands back nothing");
    ET_OK(engram_store_open(&s, NULL));
    if (!s) return;
    ET_OK(engram_store_add(s, e0, l0, 11u, 1u, &first, &na));
    ET_OK(engram_store_add(s, e1, l1, 12u, 2u, &first, &na));
    ET_OK(engram_store_serialize(s, &x, &xn));
    engram_store_close(s);
    if (!x) return;
    r1 = O_REC + R_TEXT + l0;
    ET_EQ_U64(xn, (uint64_t)(O_REC + 2u * R_TEXT + l0 + l1));       /* the layout this test patches */
    ET_OK(try_payload(x, xn));                                        /* R6: the untouched payload loads */
    m = (uint8_t *)engram_malloc(xn + 1u);
    if (!m) { engram_free(x); return; }
    {
        const patch P[] = {
            { O_VER, 4, 0u, ENGRAM_E_FORMAT, "version 0" },
            { O_VER, 4, 2u, ENGRAM_E_VERSION, "version 2" },
            { O_MAXEP, 8, 0u, ENGRAM_E_FORMAT, "max_episodes 0" },
            { O_MAXEP, 8, 1u, ENGRAM_E_FORMAT, "max_episodes under live" },
            { O_MAXTB, 8, 0u, ENGRAM_E_FORMAT, "max_text_bytes 0" },
            { O_MAXTB, 8, 50u, ENGRAM_E_FORMAT, "max_text_bytes under the text" },
            { O_CAP, 4, 0u, ENGRAM_E_FORMAT, "chunk_cap 0" },
            { O_CAP, 4, ENGRAM_EPI_TAIL + 1u, ENGRAM_E_FORMAT, "chunk_cap over the tail" },
            { O_CAP, 4, 20u, ENGRAM_E_FORMAT, "chunk_cap under the minimum" },
            { O_CAP, 4, 36u, ENGRAM_E_FORMAT, "a valid chunk_cap under an episode's length" },
            { O_OVL, 4, 300u, ENGRAM_E_FORMAT, "overlap over half the cap" },
            { O_C, 4, 0u, ENGRAM_E_FORMAT, "recall_c 0" },
            { O_RANK, 4, 3u, ENGRAM_E_FORMAT, "rank 3" },
            { O_W, 4, 0x7FC00000u, ENGRAM_E_FORMAT, "a NaN weight" },
            { O_W + 4, 4, 0x7F800000u, ENGRAM_E_FORMAT, "an infinite weight" },
            { O_W, 4, 0xBF800000u, ENGRAM_E_FORMAT, "a negative weight" },
            { O_PAIRS, 4, 2u, ENGRAM_E_FORMAT, "ordered_pairs 2" },
            { O_FOLD, 4, 3u, ENGRAM_E_FORMAT, "fold 3" },
            { O_UTF8, 4, 2u, ENGRAM_E_FORMAT, "utf8 policy 2" },
            { O_TF, 4, 4u, ENGRAM_E_FORMAT, "tf 4" },
            { O_TF, 4, 0u, ENGRAM_E_FORMAT, "tf LINEAR: no exact stage" },
            { O_NEXT, 8, 0u, ENGRAM_E_FORMAT, "next_id 0" },
            { O_NEXT, 8, 2u, ENGRAM_E_FORMAT, "next_id at a live id" },
            { O_LIVE, 8, 3u, ENGRAM_E_FORMAT, "live one too many" },
            { O_LIVE, 8, 1u, ENGRAM_E_FORMAT, "live one too few (trailing bytes)" },
            { O_LIVE, 8, (uint64_t)1u << 40, ENGRAM_E_FORMAT, "live 2^40" },
            { O_LIVE, 8, ~(uint64_t)0, ENGRAM_E_FORMAT, "live 2^64 - 1" },
            { O_REC + R_ID, 8, 0u, ENGRAM_E_FORMAT, "id 0" },
            { O_REC + R_ID, 8, 2u, ENGRAM_E_FORMAT, "ids not increasing" },
            { O_REC + R_FLAGS, 4, 0u, ENGRAM_E_FORMAT, "flags 0 (a tombstone)" },
            { O_REC + R_FLAGS, 4, 2u, ENGRAM_E_FORMAT, "flags CONSOLIDATED without LIVE" },
            { O_REC + R_FLAGS, 4, 5u, ENGRAM_E_FORMAT, "an unknown flag" },
            { O_REC + R_LEN, 4, 0u, ENGRAM_E_FORMAT, "len 0" },
            { O_REC + R_LEN, 4, ENGRAM_EPI_TAIL + 1u, ENGRAM_E_FORMAT, "len over the cap" },
            { O_REC + R_LEN, 4, 0xFFFFFFFFu, ENGRAM_E_FORMAT, "len 2^32 - 1" },
            { O_REC + R_LEN, 4, 5u, ENGRAM_E_FORMAT, "len short: the rest misparses" },
        };
        unsigned k;
        bad = 0;
        for (k = 0; k < sizeof P / sizeof P[0]; k++) {
            engram_rc rc;
            engram_alloc_stats a0, a;
            memcpy(m, x, xn);
            put(m + P[k].off, P[k].width, P[k].v);
            engram_alloc_reset_run();
            engram_alloc_stats_get(&a0);
            rc = try_payload(m, xn);
            engram_alloc_stats_get(&a);
            ET_CHECKF(rc == P[k].want, "%s: got %s", P[k].what, engram_rcname(rc));
            ET_CHECKF(a.peak_bytes - a0.live_bytes < 1024u * 1024u, "%s: peak %llu B above the start", P[k].what,
                      (unsigned long long)(a.peak_bytes - a0.live_bytes));
            if (rc != P[k].want) bad++;
        }
        printf("       %u doctored fields, each refused with the expected code\n", (unsigned)(sizeof P / sizeof P[0]));
    }
    /* cases that need more than one field changed */
    memcpy(m, x, xn);
    put(m + O_W, 4, 0u); put(m + O_W + 4, 4, 0u); put(m + O_W + 16, 4, 0u); put(m + O_W + 20, 4, 0u);
    ET_RC(try_payload(m, xn), ENGRAM_E_FORMAT);                                   /* every weight 0 */
    memcpy(m, x, xn);
    put(m + r1 + R_ID, 8, 50u);                                                    /* id at or past next_id */
    ET_RC(try_payload(m, xn), ENGRAM_E_FORMAT);
    memcpy(m, x, xn);
    memset(m + O_REC + R_TEXT, '\'', l0);                                          /* text with no feature */
    ET_RC(try_payload(m, xn), ENGRAM_E_FORMAT);
    memcpy(m, x, xn);
    m[O_REC + R_TEXT + 4] = 0xC3u; m[O_REC + R_TEXT + 5] = 0x28u;                  /* ill-formed UTF-8 */
    ET_OK(try_payload(m, xn));                                                     /* REPLACE: accepted */
    put(m + O_UTF8, 4, (uint64_t)ENGRAM_UTF8_STRICT);
    ET_RC(try_payload(m, xn), ENGRAM_E_FORMAT);                                    /* STRICT: refused   */
    memcpy(m, x, xn); m[xn] = 0u;
    ET_RC(try_payload(m, xn + 1u), ENGRAM_E_FORMAT);                               /* a trailing byte */

    ET_SECTION("F7: every truncation refused; every single-byte change refused or loaded whole -- never a crash");
    bad = 0;
    for (i = 0; i < xn; i++) if (try_payload(x, i) == ENGRAM_OK) bad++;
    ET_EQ_U64(bad, 0u);
    {
        size_t ok = 0, refused = 0, internal = 0;
        unsigned d;
        for (i = 0; i < xn; i++)
            for (d = 1; d < 256u; d += 85u) {                  /* three different values per byte */
                engram_rc rc;
                memcpy(m, x, xn);
                m[i] = (uint8_t)(m[i] ^ d);
                rc = try_payload(m, xn);
                if (rc == ENGRAM_OK) ok++;
                else if (rc == ENGRAM_E_INTERNAL) internal++;
                else refused++;
            }
        ET_EQ_U64(internal, 0u);
        printf("       %zu changed payloads: %zu refused, %zu loaded (a time, source, recall count, seed, weight\n"
               "       or text byte that is still a valid value); none crashed, none leaked\n", ok + refused, refused, ok);
    }

    ET_SECTION("F7: every allocation of serialize, deserialize, save and load failed in turn -- E_MEM, nothing leaked");
    {
        uint64_t calls, k;
        engram_store *l = NULL;
        engram_key key;
        memset(key.k, 3, 32u);
        engram_alloc_reset_run();
        ET_OK(try_payload(x, xn));
        calls = engram_alloc_calls();
        for (k = 1; k <= calls; k++) {
            engram_alloc_reset_run();
            engram_alloc_fail_at(k);
            ET_RC(try_payload(x, xn), ENGRAM_E_MEM);
            engram_alloc_fail_at(0u);
        }
        ET_OK(engram_store_deserialize(&l, x, xn));
        if (l) {
            uint8_t *y = NULL;
            size_t yn = 0;
            uint64_t fp = engram_store_fingerprint(l);
            engram_alloc_reset_run();
            ET_OK(engram_store_save(l, DIR "/f.engram", &key));
            calls = engram_alloc_calls();
            for (k = 1; k <= calls; k++) {
                engram_store *t = (engram_store *)1;
                engram_alloc_reset_run();
                engram_alloc_fail_at(k);
                ET_RC(engram_store_save(l, DIR "/f.engram", &key), ENGRAM_E_MEM);
                engram_alloc_fail_at(0u);
                ET_OK(engram_store_load(&t, DIR "/f.engram", &key));             /* the file still loads */
                if (t) { ET_EQ_U64(engram_store_fingerprint(t), fp); engram_store_close(t); }
            }
            engram_alloc_reset_run();
            {
                engram_store *t = NULL;
                ET_OK(engram_store_load(&t, DIR "/f.engram", &key));
                engram_store_close(t);
            }
            calls = engram_alloc_calls();
            for (k = 1; k <= calls; k++) {
                engram_store *t = (engram_store *)1;
                engram_alloc_reset_run();
                engram_alloc_fail_at(k);
                ET_RC(engram_store_load(&t, DIR "/f.engram", &key), ENGRAM_E_MEM);
                engram_alloc_fail_at(0u);
                ET_CHECK(t == NULL);
            }
            engram_alloc_reset_run();
            ET_OK(engram_store_serialize(l, &y, &yn));
            engram_free(y); y = NULL;
            calls = engram_alloc_calls();
            for (k = 1; k <= calls; k++) {
                engram_alloc_reset_run();
                engram_alloc_fail_at(k);
                ET_RC(engram_store_serialize(l, &y, &yn), ENGRAM_E_MEM);
                engram_alloc_fail_at(0u);
                ET_CHECK(y == NULL && yn == 0u);
            }
            ET_EQ_U64(engram_store_fingerprint(l), fp);

            ET_SECTION("F7: every IO operation of a store save over an older one failed in turn -- old or new, never torn");
            {
                engram_store *older = NULL;
                uint64_t ops, fo, t0 = 0;
                unsigned torn = 0, silent = 0, newer = 0;
                const char *t = "one more memory, so the new file differs from the old";
                size_t n1;
                ET_OK(engram_store_deserialize(&older, x, xn));
                fo = engram_store_fingerprint(older);
                ET_OK(engram_store_add(l, t, strlen(t), 77u, 0u, &t0, &n1));
                fp = engram_store_fingerprint(l);
                ET_CHECK(fp != fo);
                engram_io_reset();
                ET_OK(engram_store_save(l, DIR "/f.engram", &key));
                ops = engram_io_ops();
                for (k = 1; k <= ops; k++) {
                    engram_store *got = NULL;
                    engram_rc rc;
                    ET_OK(engram_store_save(older, DIR "/f.engram", &key));
                    engram_io_reset();
                    engram_io_fail_at(k);
                    rc = engram_store_save(l, DIR "/f.engram", &key);
                    engram_io_fail_at(0u);
                    if (rc == ENGRAM_OK) silent++;
                    if (engram_store_load(&got, DIR "/f.engram", &key) != ENGRAM_OK) { torn++; continue; }
                    if (engram_store_fingerprint(got) == fp) newer++;
                    else if (engram_store_fingerprint(got) != fo) torn++;
                    engram_store_close(got);
                }
                ET_EQ_U64(torn, 0u);
                ET_EQ_U64(silent, 0u);
                ET_EQ_U64(count_orphans(), 0u);
                printf("       %llu IO operations per save, each failed: 0 torn, 0 unreported, %u after the commit point\n",
                       (unsigned long long)ops, newer);
                engram_store_close(older);
            }
            ET_OK(engram_file_remove(DIR "/f.engram"));
            engram_store_close(l);
        }
    }
    engram_free(m);
    engram_free(x);
}

/* ==================================================================================================
 * F8: the router
 * ============================================================================================== */
static void rand_unit(engram_rng *r, float *v, unsigned d)
{
    double s = 0.0;
    unsigned i;
    for (i = 0; i < d; i++) {
        double x = 0.0;
        unsigned k;
        for (k = 0; k < 4u; k++) x += (double)(engram_rng_u64(r) >> 11) * (1.0 / 9007199254740992.0) - 0.5;
        v[i] = (float)x;
        s += x * x;
    }
    s = sqrt(s);
    for (i = 0; i < d; i++) v[i] = (float)((double)v[i] / s);
}

static void clustered(engram_rng *r, const float *cen, unsigned nc, float *v, unsigned d)
{
    const float *c = cen + (size_t)engram_rng_below(r, nc) * d;
    double s = 0.0;
    unsigned i;
    rand_unit(r, v, d);
    for (i = 0; i < d; i++) { double x = (double)c[i] + 0.6 * (double)v[i]; v[i] = (float)x; s += x * x; }
    s = sqrt(s);
    for (i = 0; i < d; i++) v[i] = (float)((double)v[i] / s);
}

static engram_router *build_router(unsigned d, size_t n, unsigned C, int train, uint64_t seed)
{
    engram_router_cfg cfg;
    engram_router *r = NULL;
    engram_rng g;
    float *v, cen[16 * ENGRAM_D];
    size_t i;
    unsigned nc = 16u;
    engram_router_cfg_default(&cfg);
    cfg.dim = d;
    cfg.iters = 6u;
    if (engram_router_open(&r, &cfg) != ENGRAM_OK) return NULL;
    v = (float *)engram_array(d, sizeof *v);
    if (!v) { engram_router_close(r); return NULL; }
    engram_rng_seed(&g, seed);
    for (i = 0; i < nc; i++) rand_unit(&g, cen + i * d, d);
    for (i = 0; i < n; i++) { clustered(&g, cen, nc, v, d); (void)engram_router_add(r, 1000u + 3u * i, v); }
    if (train && C) {
        (void)engram_router_train(r, C);
        for (i = 0; i < n / 4u; i++) { clustered(&g, cen, nc, v, d); (void)engram_router_add(r, 500000u + i, v); }
        for (i = 0; i < n; i += 5u) (void)engram_router_remove(r, 1000u + 3u * i);
    }
    engram_free(v);
    return r;
}

static void router_answers_equal(engram_router *a, engram_router *b, unsigned d, const char *what)
{
    engram_rng g;
    float q[ENGRAM_D];
    size_t i, bad = 0;
    engram_rng_seed(&g, 0xA115u);
    for (i = 0; i < 40u; i++) {
        engram_route_hit ha[8], hb[8];
        engram_route_cost ca, cb;
        size_t na = 0, nb = 0;
        unsigned np = (unsigned)(i % 4u);
        rand_unit(&g, q, d);
        if (engram_router_search(a, q, 8u, np, ha, &na, &ca) != ENGRAM_OK ||
            engram_router_search(b, q, 8u, np, hb, &nb, &cb) != ENGRAM_OK ||
            na != nb || memcmp(ha, hb, na * sizeof ha[0]) != 0 || memcmp(&ca, &cb, sizeof ca) != 0) bad++;
        if (engram_router_exact(a, q, 8u, ha, &na, NULL) != ENGRAM_OK ||
            engram_router_exact(b, q, 8u, hb, &nb, NULL) != ENGRAM_OK ||
            na != nb || memcmp(ha, hb, na * sizeof ha[0]) != 0) bad++;
    }
    ET_CHECKF(bad == 0u, "%s: %zu searches differ", what, bad);
}

static void roundtrip_router(engram_router *r, const engram_key *key, unsigned d, const char *what)
{
    engram_router *l = NULL;
    engram_router_stats x, y;
    uint8_t *a = NULL, *b = NULL;
    size_t an = 0, bn = 0;
    uint64_t fp;
    {   /* searches first, so the search counter has something to carry across */
        engram_rng g;
        float q[ENGRAM_D];
        engram_route_hit h[2];
        size_t nh = 0;
        unsigned k;
        engram_rng_seed(&g, 0x5EA7u);
        for (k = 0; k < 3u; k++) {
            rand_unit(&g, q, d);
            ET_OK(engram_router_search(r, q, 2u, 0u, h, &nh, NULL));
        }
    }
    fp = engram_router_fingerprint(r);
    ET_OK(engram_router_save(r, DIR "/router.engram", key));
    ET_EQ_U64(engram_router_fingerprint(r), fp);
    ET_OK(engram_router_load(&l, DIR "/router.engram", key));
    if (!l) return;
    ET_EQ_U64(engram_router_fingerprint(l), fp);
    ET_OK(engram_router_check(l));
    engram_router_stats_get(r, &x);
    engram_router_stats_get(l, &y);
    ET_CHECK(x.searches >= 3u);
    ET_CHECKF(x.keys == y.keys && x.buckets == y.buckets && x.empty_buckets == y.empty_buckets &&
              x.largest_bucket == y.largest_bucket && x.trained == y.trained && x.adds == y.adds &&
              x.removes == y.removes && x.trains == y.trains && x.searches == y.searches, "%s: stats differ", what);
    ET_OK(engram_router_serialize(r, &a, &an));
    ET_OK(engram_router_serialize(l, &b, &bn));
    ET_CHECKF(a && b && an == bn && memcmp(a, b, an) == 0, "%s: serialize(load(x)) != x", what);
    engram_free(a); engram_free(b);
    router_answers_equal(r, l, d, what);
    {   /* it keeps working: the same add and remove on both leave them equal */
        engram_rng g;
        float v[ENGRAM_D];
        engram_rng_seed(&g, 0xADDu);
        rand_unit(&g, v, d);
        ET_OK(engram_router_add(r, 0xFEEDu, v));
        ET_OK(engram_router_add(l, 0xFEEDu, v));
        ET_EQ_U64(engram_router_fingerprint(l), engram_router_fingerprint(r));
        ET_OK(engram_router_remove(r, 0xFEEDu));
        ET_OK(engram_router_remove(l, 0xFEEDu));
        ET_OK(engram_router_check(l));
    }
    engram_router_close(l);
}

static engram_rc try_router(const uint8_t *p, size_t n)
{
    engram_router *r = (engram_router *)1;
    engram_rc rc = engram_router_deserialize(&r, p, n);
    if (rc == ENGRAM_OK) {
        if (engram_router_check(r) != ENGRAM_OK) rc = ENGRAM_E_INTERNAL;
        engram_router_close(r);
    } else if (r != NULL) return ENGRAM_E_INTERNAL;
    return rc;
}

/* router payload layout (engram_router.c): 80 fixed bytes, then C x dim floats if trained, then keys */
enum { RO_VER = 0, RO_DIM = 4, RO_NPROBE = 8, RO_ITERS = 12, RO_TMAX = 16, RO_SEED = 24, RO_TRAINED = 32,
       RO_C = 36, RO_N = 72, RO_BODY = 80 };

static void test_router_persist(void)
{
    engram_key key;
    engram_router *r;
    unsigned i;
    for (i = 0; i < 32u; i++) key.k[i] = (uint8_t)(0x33u + i);

    ET_SECTION("F8: routers saved and loaded -- untrained, trained, grown and shrunk after training; empty");
    r = build_router(64u, 300u, 0u, 0, 0x1u);
    if (r) { roundtrip_router(r, NULL, 64u, "untrained"); engram_router_close(r); }
    r = build_router(64u, 600u, 12u, 1, 0x2u);
    if (r) {
        engram_router_stats st;
        engram_router_stats_get(r, &st);
        ET_CHECK(st.trained && st.buckets == 12u && st.removes > 0u);
        roundtrip_router(r, &key, 64u, "trained");
        engram_router_close(r);
    }
    r = build_router(256u, 400u, 20u, 1, 0x3u);
    if (r) { roundtrip_router(r, &key, 256u, "trained d256"); engram_router_close(r); }
    r = NULL;
    ET_OK(engram_router_open(&r, NULL));
    if (r) { roundtrip_router(r, NULL, ENGRAM_D, "empty"); engram_router_close(r); }
    r = build_router(64u, 40u, 4u, 1, 0x4u);                     /* trained, then every key removed */
    if (r) {
        engram_router_stats st;
        uint8_t *x = NULL;
        size_t xn = 0;
        (void)engram_router_serialize(r, &x, &xn);
        engram_router_close(r);
        r = NULL;
        ET_OK(engram_router_deserialize(&r, x, xn));
        engram_free(x);
        if (r) {
            size_t k;
            for (k = 0; k < 40u; k++) (void)engram_router_remove(r, 1000u + 3u * k);
            for (k = 0; k < 10u; k++) (void)engram_router_remove(r, 500000u + k);
            engram_router_stats_get(r, &st);
            ET_CHECK(st.keys == 0u && st.trained);
            roundtrip_router(r, &key, 64u, "trained, emptied");
            engram_router_close(r);
        }
    }

    ET_SECTION("F8: the loader refuses every doctored field; truncations; a key placed nowhere near its centroid is re-placed");
    r = build_router(64u, 50u, 4u, 1, 0x5u);
    if (r) {
        uint8_t *x = NULL, *m = NULL;
        size_t xn = 0, k, bad = 0, k0 = RO_BODY + 4u * 64u * 4u;
        ET_OK(engram_router_serialize(r, &x, &xn));
        m = (uint8_t *)engram_malloc(xn + 1u);
        if (x && m) {
            const patch P[] = {
                { RO_VER, 4, 0u, ENGRAM_E_FORMAT, "version 0" },
                { RO_VER, 4, 9u, ENGRAM_E_VERSION, "version 9" },
                { RO_DIM, 4, 0u, ENGRAM_E_FORMAT, "dim 0" },
                { RO_DIM, 4, 60u, ENGRAM_E_FORMAT, "dim not a multiple of 8" },
                { RO_DIM, 4, 4104u, ENGRAM_E_FORMAT, "dim over 4096" },
                { RO_DIM, 4, 128u, ENGRAM_E_FORMAT, "dim 128: the sizes no longer add up" },
                { RO_NPROBE, 4, 0u, ENGRAM_E_FORMAT, "nprobe 0" },
                { RO_ITERS, 4, 0u, ENGRAM_E_FORMAT, "iters 0" },
                { RO_TRAINED, 4, 2u, ENGRAM_E_FORMAT, "trained 2" },
                { RO_TRAINED, 4, 0u, ENGRAM_E_FORMAT, "untrained with 4 buckets" },
                { RO_C, 4, 0u, ENGRAM_E_FORMAT, "C 0" },
                { RO_C, 4, 5u, ENGRAM_E_FORMAT, "C one too many" },
                { RO_C, 4, 0xFFFFFFFFu, ENGRAM_E_FORMAT, "C 2^32 - 1" },
                { RO_N, 8, (uint64_t)1u << 40, ENGRAM_E_FORMAT, "n 2^40" },
                { RO_N, 8, 1u, ENGRAM_E_FORMAT, "n too small (trailing bytes)" },
                { RO_BODY, 4, 0x7FC00000u, ENGRAM_E_FORMAT, "a NaN in a centroid" },
                { RO_BODY + 4, 4, 0x40000000u, ENGRAM_E_FORMAT, "a centroid off the unit sphere" },
                { 0, 8, 0u, ENGRAM_E_FORMAT, "key id 0" },
                { 8, 4, 0x7F800000u, ENGRAM_E_FORMAT, "an infinite key component" },
                { 12, 4, 0x3F800000u, ENGRAM_E_FORMAT, "a key off the unit sphere" },
            };
            unsigned q;
            ET_OK(try_router(x, xn));                                          /* R6 */
            for (q = 0; q < sizeof P / sizeof P[0]; q++) {
                engram_rc rc;
                engram_alloc_stats a0, a;
                unsigned off = P[q].off + (q >= 17u ? (unsigned)k0 : 0u);
                memcpy(m, x, xn);
                put(m + off, P[q].width, P[q].v);
                engram_alloc_reset_run();
                engram_alloc_stats_get(&a0);
                rc = try_router(m, xn);
                engram_alloc_stats_get(&a);
                ET_CHECKF(rc == P[q].want, "%s: got %s", P[q].what, engram_rcname(rc));
                ET_CHECKF(a.peak_bytes - a0.live_bytes < 1024u * 1024u, "%s: peak %llu B above the start", P[q].what,
                          (unsigned long long)(a.peak_bytes - a0.live_bytes));
            }
            printf("       %u doctored fields, each refused with the expected code\n", (unsigned)(sizeof P / sizeof P[0]));
            /* a duplicated id: the second key record carries the first's id */
            memcpy(m, x, xn);
            memcpy(m + k0 + (8u + 64u * 4u), m + k0, 8u);
            ET_RC(try_router(m, xn), ENGRAM_E_FORMAT);
            memcpy(m, x, xn); m[xn] = 0u;
            ET_RC(try_router(m, xn + 1u), ENGRAM_E_FORMAT);
            for (k = 0; k < xn; k++) if (try_router(x, k) == ENGRAM_OK) bad++;
            ET_EQ_U64(bad, 0u);
            /* a key's vector swapped for another centroid's neighbourhood: it loads, and into THAT bucket */
            memcpy(m, x, xn);
            memcpy(m + k0 + 8u, m + RO_BODY + 3u * 64u * 4u, 64u * 4u);          /* key 0 := centroid 3 */
            ET_OK(try_router(m, xn));
            {
                engram_router *t = NULL;
                engram_route_hit h[1];
                size_t nh = 0;
                engram_route_cost c;
                ET_OK(engram_router_deserialize(&t, m, xn));
                if (t) {
                    float cq[64];
                    memcpy(cq, m + RO_BODY + 3u * 64u * 4u, sizeof cq);
                    ET_OK(engram_router_search(t, cq, 1u, 1u, h, &nh, &c));        /* one probe: bucket 3 */
                    ET_CHECK(nh == 1u && h[0].id == engram_le_get_u64(m + k0));
                    engram_router_close(t);
                }
            }
        }
        {
            uint64_t calls, kk;
            engram_alloc_reset_run();
            ET_OK(try_router(x, xn));
            calls = engram_alloc_calls();
            for (kk = 1; kk <= calls; kk++) {
                engram_router *t = (engram_router *)1;
                engram_rc rc;
                engram_alloc_reset_run();
                engram_alloc_fail_at(kk);
                rc = engram_router_deserialize(&t, x, xn);
                engram_alloc_fail_at(0u);
                ET_RC(rc, ENGRAM_E_MEM);
                ET_CHECK(t == NULL);
            }
            printf("       every one of %llu allocations of a load failed in turn: E_MEM, nothing leaked\n",
                   (unsigned long long)calls);
        }
        engram_free(m); engram_free(x);
        engram_router_close(r);
    }

    ET_SECTION("F8: what a load costs -- C dot products per key to re-place it");
    {
        size_t n = g_quick ? 1024u : 4096u;
        unsigned C = g_quick ? 32u : 64u;
        uint64_t t0;
        uint8_t *x = NULL;
        size_t xn = 0;
        r = build_router(ENGRAM_D, n, C, 1, 0x6u);
        if (r) {
            engram_router *l = NULL;
            ET_OK(engram_router_serialize(r, &x, &xn));
            t0 = engram_now_ns();
            ET_OK(engram_router_deserialize(&l, x, xn));
            printf("       %zu keys, d %u, C %u: %.1f ms to load (%.2f MB)\n", n + n / 4u - n / 5u, ENGRAM_D, C,
                   ms_since(t0), (double)xn / 1e6);
            if (l) { ET_EQ_U64(engram_router_fingerprint(l), engram_router_fingerprint(r)); engram_router_close(l); }
            engram_free(x);
            engram_router_close(r);
        }
    }
    (void)engram_file_remove(DIR "/router.engram");
}

/* ==================================================================================================
 * F9: scale, and the print the gate compares across platforms
 * ============================================================================================== */
static void test_scale_and_print(void)
{
    engram_store *s = NULL, *l = NULL;
    engram_key key;
    uint8_t *raw = NULL, *x = NULL, *f = NULL, salt[32];
    size_t rawn = 0, xn = 0, fn = 0, i, start = 0, paras = 0, limit;
    char *path;
    uint64_t t0, first, fp;
    size_t na;
    for (i = 0; i < 32u; i++) { key.k[i] = (uint8_t)(0xC0u ^ i); salt[i] = (uint8_t)(i * 5u); }

    ET_SECTION("F9: the scale corpus saved and loaded, sealed -- every episode re-signed, the same fingerprint");
    path = engram_path_join(data_dir(), "scale_docs.txt");
    if (!path || engram_file_read(path, &raw, &rawn) != ENGRAM_OK) {
        ET_CHECKF(0, "scale_docs.txt did not load");
        engram_free(path);
        return;
    }
    engram_free(path);
    ET_OK(engram_store_open(&s, NULL));
    limit = g_quick ? 1500u : (size_t)-1;
    for (i = 0; s && i <= rawn && paras < limit; i++) {
        if (i == rawn || raw[i] == '\n') {
            if (i > start) { (void)engram_store_add(s, raw + start, i - start, (uint64_t)paras, 0u, &first, &na); paras++; }
            start = i + 1u;
        }
    }
    if (s) {
        engram_store_stats st;
        engram_store_stats_get(s, &st);
        fp = engram_store_fingerprint(s);
        t0 = engram_now_ns();
        ET_OK(engram_store_save(s, DIR "/scale.engram", &key));
        printf("       %zu episodes (%.1f MB of text): saved sealed in %.0f ms\n", st.live, (double)st.text_bytes / 1e6,
               ms_since(t0));
        t0 = engram_now_ns();
        ET_OK(engram_store_load(&l, DIR "/scale.engram", &key));
        printf("       loaded -- every signature recomputed -- in %.0f ms\n", ms_since(t0));
        if (l) { ET_EQ_U64(engram_store_fingerprint(l), fp); engram_store_close(l); }
        ET_OK(engram_file_remove(DIR "/scale.engram"));
        engram_store_close(s);
    }
    engram_free(raw);

    ET_SECTION("F9: PERSISTPRINT -- a store and a router, serialized and sealed with a fixed salt: the file's bits");
    {
        engram_store_cfg cfg;
        engram_router *r;
        uint64_t h = 0;
        engram_store_cfg_default(&cfg);
        s = build_store(&cfg, 120u);
        if (s) {
            ET_OK(engram_store_serialize(s, &x, &xn));
            ET_OK(engram_seal_pack(ENGRAM_KIND_STORE, &key, salt, x, xn, &f, &fn));
            if (f) h = engram_mix2(h, sha_print(f, fn));
            engram_free(x); engram_free(f); x = NULL; f = NULL;
            engram_store_close(s);
        }
        r = build_router(64u, 500u, 10u, 1, 0x77u);
        if (r) {
            ET_OK(engram_router_serialize(r, &x, &xn));
            ET_OK(engram_seal_pack(ENGRAM_KIND_ROUTER, &key, salt, x, xn, &f, &fn));
            if (f) h = engram_mix2(h, sha_print(f, fn));
            engram_free(x); engram_free(f);
            engram_router_close(r);
        }
        printf("PERSISTPRINT %016llX\n", (unsigned long long)h);
    }
}

int main(void)
{
    const char *q = getenv("ENGRAM_TEST_QUICK");
    g_quick = q && q[0] == '1';
    printf("ENGRAM P1.5 -- the container, the keyfile, and persistence\n");
    ET_SELFTEST();
    if (engram_dir_make(DIR) != ENGRAM_OK) { printf("cannot make %s\n", DIR); return 1; }
    if (!load_lines(&g_en, "corpus_en.txt", 400u) || !load_lines(&g_de, "corpus_de.txt", 40u) ||
        !load_lines(&g_zh, "corpus_zh.txt", 40u)) {
        ET_CHECKF(0, "the corpora did not load from %s", data_dir());
        return et_report("test_persist");
    }
    test_vectors();
    test_refusals();
    test_files();
    test_keyfile();
    test_store_roundtrip();
    test_store_doctored();
    test_router_persist();
    test_scale_and_print();
    engram_free(g_en.buf); engram_free(g_de.buf); engram_free(g_zh.buf);
    return et_report("test_persist");
}
