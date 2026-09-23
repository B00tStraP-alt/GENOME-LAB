/* ==================================================================================================
 * engram_seal.h -- THE CONTAINER every ENGRAM file is written in, and the key it is sealed with.
 * ==================================================================================================
 *
 * ONE FORMAT FOR EVERY FILE
 * ==================================================================================================
 * The episodic store, the router, the slow store and the diary all reach the disk through this file,
 * so there is one place where a file is proven intact -- or sealed -- and one set of tests for it.
 *
 *      offset  size  field
 *        0       8   magic  "ENGRAM" 1A 0A    (1A stops a Windows `type`; 0A catches a text-mode
 *                                               transfer that rewrote line endings -- the PNG trick)
 *        8       2   format version (1)
 *       10       2   kind            (what the payload is: store, router, ... -- a file opened as the
 *                                     wrong kind is refused, not misparsed)
 *       12       4   flags           bit 0: SEALED. Any other bit set: refused (a newer writer's
 *                                    feature this reader does not know)
 *       16       8   payload length
 *       24      32   salt            fresh random bytes for every write
 *       56       8   reserved, zero
 *       64       n   payload         plaintext, or ciphertext when sealed
 *     64+n      32   trailer         PLAIN:  SHA-256 of bytes 0 .. 64+n
 *                                    SEALED: the 16-byte Poly1305 tag, then 16 zero bytes
 *
 *   SEALED   the file key and nonce are HKDF-SHA-256(master key, salt, "ENGRAM seal v1" | kind): a
 *            new key for every write, so no (key, nonce) pair can ever repeat, however many times a
 *            file is rewritten. The 64-byte header is the AEAD's associated data: flipping the
 *            SEALED bit, the kind or the length breaks the tag. ChaCha20-Poly1305 (engram_crypto.h).
 *   PLAIN    SHA-256 detects corruption. It is not a defence against a deliberate edit -- anyone can
 *            recompute a hash -- which is why a caller holding a key REFUSES plain files: otherwise
 *            an attacker could replace a sealed store with a plain one of their own making.
 *
 * Every read proves the whole file before a byte of payload is returned; a file that fails is
 * reported, never partly loaded. Writes are atomic (engram_file_write_atomic).
 *
 * THE KEY
 * ==================================================================================================
 * The password is stretched ONCE, by Argon2id, into a 32-byte master key; every file key is derived
 * from it. The stretch parameters and salt live in a small KEYFILE (plain: nothing in it is secret)
 * with a verifier -- HMAC(master, "ENGRAM key check") -- so a wrong password is reported as a wrong
 * password, before any store is touched, and not as a corrupt store.
 * ============================================================================================== */
#ifndef ENGRAM_SEAL_H
#define ENGRAM_SEAL_H

#include "engram.h"
#include "engram_crypto.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_SEAL_VERSION   1u
#define ENGRAM_SEAL_HEADER    64u
#define ENGRAM_SEAL_TRAILER   32u
#define ENGRAM_SEAL_F_SEALED  0x1u

/* What a file holds. Never renumbered. */
#define ENGRAM_KIND_STORE     1u
#define ENGRAM_KIND_ROUTER    2u
#define ENGRAM_KIND_SLOW      3u
#define ENGRAM_KIND_DIARY     4u
#define ENGRAM_KIND_KEYFILE   5u
#define ENGRAM_KIND_TEST      99u

typedef struct {
    uint8_t k[32];
} engram_key;

void engram_key_wipe(engram_key *key);

/* ---- the container, in memory (the file functions below are these plus atomic IO) ------------ */
/* Pack a payload. key NULL: PLAIN. salt NULL: fresh random bytes (engram_os_random); a fixed salt is
 * for tests that need identical bytes on every platform. *out is engram_alloc'd. */
engram_rc engram_seal_pack(uint16_t kind, const engram_key *key, const uint8_t *salt, const void *payload,
                           size_t n, uint8_t **out, size_t *out_n);

/* Unpack and PROVE a container. On success *payload (engram_alloc'd, NUL-terminated beyond n) holds
 * the plaintext. Refusals, each leaving *payload NULL:
 *   ENGRAM_E_FORMAT   not an ENGRAM container, a length that does not add up, unknown flags, or a
 *                     kind other than the one asked for
 *   ENGRAM_E_VERSION  a format version this build does not read
 *   ENGRAM_E_STATE    SEALED but no key given
 *   ENGRAM_E_AUTH     the hash or the tag does not verify (corrupt, tampered, or the wrong key), or
 *                     a PLAIN file where a key was given (see above)
 *   ENGRAM_E_MEM */
engram_rc engram_seal_unpack(uint16_t kind, const engram_key *key, const void *file, size_t file_n,
                             uint8_t **payload, size_t *n);

engram_rc engram_seal_write(const char *path, uint16_t kind, const engram_key *key, const void *payload, size_t n);
engram_rc engram_seal_read(const char *path, uint16_t kind, const engram_key *key, uint8_t **payload, size_t *n);

/* ---- the keyfile ------------------------------------------------------------------------------ */
typedef struct {
    engram_argon2_params kdf;          /* default: 3 passes, 64 MiB, 1 lane */
} engram_keyfile_cfg;

void engram_keyfile_cfg_default(engram_keyfile_cfg *cfg);

/* Stretch the password (fresh random salt), write the keyfile, return the master key. The keyfile is
 * written atomically and NEVER over an existing one -- ENGRAM_E_EXISTS if path exists, decided by the
 * commit itself (engram_file_create_atomic), so no race with another process and no failed check can
 * replace a keyfile: its salt is the only way back to the key every sealed file is under. */
engram_rc engram_keyfile_create(const char *path, const engram_keyfile_cfg *cfg, const void *password,
                                size_t plen, engram_key *out);

/* Stretch the password with the keyfile's own salt and parameters and check it against the verifier.
 * ENGRAM_E_AUTH means the password is wrong, and nothing else: a damaged or doctored keyfile is
 * ENGRAM_E_FORMAT (a later payload version ENGRAM_E_VERSION), so the two are never confused. The
 * parameters in the file are bounded before any memory is committed to them: a doctored keyfile cannot
 * ask for a terabyte. Every keyfile function wipes *out on entry; *out holds a key only on ENGRAM_OK. */
engram_rc engram_keyfile_open(const char *path, const void *password, size_t plen, engram_key *out);

/* The same two, over a buffer: for tests and fuzzing. */
engram_rc engram_keyfile_pack(const engram_keyfile_cfg *cfg, const uint8_t salt[16], const void *password,
                              size_t plen, engram_key *out, uint8_t **file, size_t *file_n);
engram_rc engram_keyfile_unpack(const void *file, size_t file_n, const void *password, size_t plen,
                                engram_key *out);

/* The largest stretch a keyfile may ask for: 4 GiB, 64 passes, 16 lanes. */
#define ENGRAM_KDF_MAX_KIB    (4u * 1024u * 1024u)
#define ENGRAM_KDF_MAX_PASSES 64u
#define ENGRAM_KDF_MAX_LANES  16u

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_SEAL_H */
