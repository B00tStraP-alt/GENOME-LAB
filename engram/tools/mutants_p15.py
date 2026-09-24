#!/usr/bin/env python3
"""
mutants_p15.py -- the P1.5 mutation campaign: is every check in the crypto, the container, the keyfile
and the loaders actually TESTED? (COMMANDMENTS W-P1.5-8.) The runner is tools/mutants.py.

Usage:  tools/mutants_p15.py [prefix ...]      (a prefix selects mutants by name, e.g. C13 S1 P9)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mutants  # noqa: E402

M = [
 ("C1 ct_equal ignores the last byte", "engram_crypto.c", "for (i = 0; i < n; i++) d = (uint8_t)(d | (x[i] ^ y[i]));", "for (i = 0; i + 1u < n; i++) d = (uint8_t)(d | (x[i] ^ y[i]));"),
 ("C2 SHA-256 round constant", "engram_crypto.c", "0x428a2f98u", "0x428a2f99u"),
 ("C3 SHA-256 padding byte", "engram_crypto.c", "c->buf[c->used++] = 0x80u;", "c->buf[c->used++] = 0x00u;"),
 ("C4 HMAC ipad", "engram_crypto.c", "(uint8_t)(k[i] ^ 0x36u)", "(uint8_t)(k[i] ^ 0x37u)"),
 ("C5 HKDF counter from 0", "engram_crypto.c", "unsigned counter = 1;", "unsigned counter = 0;"),
 ("C6 BLAKE2b final flag word", "engram_crypto.c", "if (last) v[14] = ~v[14];", "if (last) v[15] = ~v[15];"),
 ("C7 BLAKE2b key length in the parameter block", "engram_crypto.c", "((uint64_t)keylen << 8)", "((uint64_t)keylen << 9)"),
 ("C8 Argon2id: data-independent for one slice only", "engram_crypto.c", "int indep = pass == 0u && slice < 2u;", "int indep = pass == 0u && slice < 1u;"),
 ("C9 Argon2id: type field in the address input", "engram_crypto.c", "inputb.v[5] = 2u;", "inputb.v[5] = 1u;"),
 ("C10 ChaCha20 rotation 16 -> 15", "engram_crypto.c", "d = engram_rotl32(d, 16);", "d = engram_rotl32(d, 15);"),
 ("C11 ChaCha20 block counter never advances", "engram_crypto.c", "engram_chacha20_block(key, counter++, nonce, ks);", "engram_chacha20_block(key, counter, nonce, ks);"),
 ("C12 Poly1305 clamp", "engram_crypto.c", "& 0x3ffff03u;", "& 0x3ffff07u;"),
 ("C13a Poly1305 block carry *5 -> *4", "engram_crypto.c", "        h0 += cc * 5u; cc = h0 >> 26; h0 &= 0x3ffffffu;\n        h1 += cc;\n        m += 16;", "        h0 += cc * 4u; cc = h0 >> 26; h0 &= 0x3ffffffu;\n        h1 += cc;\n        m += 16;"),
 ("C13b Poly1305 final carry *5 -> *4", "engram_crypto.c", "    h0 += cc * 5u; cc = h0 >> 26; h0 &= 0x3ffffffu;\n    h1 += cc;\n    g0", "    h0 += cc * 4u; cc = h0 >> 26; h0 &= 0x3ffffffu;\n    h1 += cc;\n    g0"),
 ("C14 Poly1305 final select", "engram_crypto.c", "g4 = h4 + cc - (1u << 26);", "g4 = h4 + cc - (1u << 25);"),
 ("C15 Poly1305 short-block marker", "engram_crypto.c", "c->buf[c->used] = 1u;", "c->buf[c->used] = 0u;"),
 ("C16 AEAD: AAD not padded", "engram_crypto.c", "engram_poly1305_update(&pc, zeros, (16u - aad_len % 16u) % 16u);", "(void)0;"),
 ("C17 AEAD: length block swapped", "engram_crypto.c", "engram_st64le(lens, (uint64_t)aad_len);", "engram_st64le(lens, (uint64_t)n);"),
 ("C18 AEAD open: output not wiped on failure", "engram_crypto.c", "if (pt && pt != ct) engram_wipe(pt, n);", "(void)0;"),
 ("C19 AEAD open: tag compared on 15 bytes", "engram_crypto.c", "if (!engram_ct_equal(want, tag, 16u)) {", "if (!engram_ct_equal(want, tag, 15u)) {"),
 ("C20 AEAD seal: keystream from block 0", "engram_crypto.c", "engram_chacha20_xor(key, 1u, nonce, pt, ct, n);", "engram_chacha20_xor(key, 0u, nonce, pt, ct, n);"),
 ("S1 container: kind not checked", "engram_seal.c", "if (k != kind || (flags & ~(uint32_t)ENGRAM_SEAL_F_SEALED) != 0u) return ENGRAM_E_FORMAT;", "(void)kind; if ((flags & ~(uint32_t)ENGRAM_SEAL_F_SEALED) != 0u) return ENGRAM_E_FORMAT;"),
 ("S2 container: plain accepted with a key (downgrade)", "engram_seal.c", "if (!(flags & ENGRAM_SEAL_F_SEALED) && key) return ENGRAM_E_AUTH;", "(void)0;"),
 ("S3 container: reserved bytes not checked", "engram_seal.c", "for (i = 56; i < ENGRAM_SEAL_HEADER; i++) if (f[i]) return ENGRAM_E_FORMAT;", "(void)0;"),
 ("S4 container: kind not bound into the file key", "engram_seal.c", "info[14] = (uint8_t)kind;", "info[14] = 0u;"),
 ("S5 container: header not authenticated", "engram_seal.c", "engram_aead_seal(okm, okm + 32, buf, ENGRAM_SEAL_HEADER,", "engram_aead_seal(okm, okm + 32, buf, 0u,"),
 ("S6 keyfile: verifier not compared", "engram_seal.c", "if (!engram_ct_equal(ver_file, ver_now, 32u)) { engram_key_wipe(out); return ENGRAM_E_AUTH; }", "(void)ver_file;"),
 ("S7 keyfile: lanes not bounded", "engram_seal.c", "p->lanes <= ENGRAM_KDF_MAX_LANES &&", ""),
 ("S8 container: sealed trailer padding not checked", "engram_seal.c", "if (f[ENGRAM_SEAL_HEADER + len + 16u + i]) { engram_free(pt); return ENGRAM_E_FORMAT; }", "(void)0;"),
 ("S9 keyfile: damage reported as a wrong password", "engram_seal.c", "if (rc == ENGRAM_E_AUTH) return ENGRAM_E_FORMAT;", "(void)0;"),
 ("S10 container: version not checked", "engram_seal.c", "if (ver != ENGRAM_SEAL_VERSION) return ENGRAM_E_VERSION;", "(void)0;"),
 ("S11 keyfile: key not wiped on entry to open", "engram_seal.c", "    if (out) engram_key_wipe(out);\n    if (!path || !out) return ENGRAM_E_ARG;\n    rc = engram_file_read", "    if (!path || !out) return ENGRAM_E_ARG;\n    rc = engram_file_read"),
 ("P1 store load: ids need not increase", "engram_store.c", "if (id == 0u || id <= prev || id >= next_id ||", "if (id == 0u || ((void)prev, 0) || id >= next_id ||"),
 ("P2 store load: any flags", "engram_store.c", "(flags != ENGRAM_EPI_LIVE && flags != (ENGRAM_EPI_LIVE | ENGRAM_EPI_CONSOLIDATED)) ||", "(flags > 3u) ||"),
 ("P3 store load: evictions counter dropped", "engram_store.c", "s->adds = adds; s->deletes = deletes; s->evictions = evictions;", "s->adds = adds; s->deletes = deletes; (void)evictions;"),
 ("P4 store load: recall counts dropped", "engram_store.c", "s->recalls[i] = engram_rbuf_u32(&r);", "(void)engram_rbuf_u32(&r); s->recalls[i] = 0u;"),
 ("P5 store load: a text that does not encode accepted", "engram_store.c", "if (rc != ENGRAM_OK) { engram_store_close(s); return ENGRAM_E_FORMAT; }", "(void)rc;"),
 ("P6 store load: consolidated count not rebuilt", "engram_store.c", "if (flags & ENGRAM_EPI_CONSOLIDATED) cons++;", "(void)0;"),
 ("P7 store load: live not bounded by the bytes left", "engram_store.c", "live > engram_rbuf_left(&r) / ENGRAM_STORE_REC_MIN", "0"),
 ("P8 store save: flags written as LIVE", "engram_store.c", "engram_wbuf_u32(&w, s->flags[slot]);", "engram_wbuf_u32(&w, ENGRAM_EPI_LIVE);"),
 ("P9 store load: text length not bounded by the cap", "engram_store.c", "len == 0u || len > cfg.chunk_cap ||", "len == 0u ||"),
 ("P10 store load: times dropped", "engram_store.c", "s->time_ms[i] = engram_rbuf_u64(&r);", "(void)engram_rbuf_u64(&r); s->time_ms[i] = 0u;"),
 ("P11 store load: configuration not checked", "engram_store.c", "if (engram_store_cfg_check(&cfg) != ENGRAM_OK) return ENGRAM_E_FORMAT;", "(void)0;"),
 ("R1 router load: keys not proven unit", "engram_router.c", "if (id == 0u || !engram_unit_ok(r->key + i * dim, (unsigned)dim) || engram_hfind(r, id) != (size_t)-1) goto fail;", "if (id == 0u || engram_hfind(r, id) != (size_t)-1) goto fail;"),
 ("R2 router load: duplicate ids accepted", "engram_router.c", "if (id == 0u || !engram_unit_ok(r->key + i * dim, (unsigned)dim) || engram_hfind(r, id) != (size_t)-1) goto fail;", "if (id == 0u || !engram_unit_ok(r->key + i * dim, (unsigned)dim)) goto fail;"),
 ("R3 router load: buckets not recomputed", "engram_router.c", "r->bkt[i] = engram_nearest(r, r->key + i * dim);", "r->bkt[i] = 0u;"),
 ("R4 router load: search counter dropped", "engram_router.c", "r->adds = adds; r->removes = removes; r->trains = trains; r->searches = searches;", "r->adds = adds; r->removes = removes; r->trains = trains; (void)searches;"),
 ("R5 router load: centroids not proven unit", "engram_router.c", "for (c = 0; c < C; c++) if (!engram_unit_ok(r->cen + (size_t)c * dim, (unsigned)dim)) goto fail;", "(void)0;"),
 ("R6 router load: exact length not required", "engram_router.c", "if (engram_rbuf_left(&rb) != (trained ? (size_t)C * dim * 4u : 0u) + (size_t)nk * rec) return ENGRAM_E_FORMAT;", ";"),
 ("R7 router save: always marked trained", "engram_router.c", "engram_wbuf_u32(&w, r->trained ? 1u : 0u);", "engram_wbuf_u32(&w, 1u);"),
 ("R8 router load: id 0 accepted", "engram_router.c", "if (id == 0u || !engram_unit_ok(", "if (!engram_unit_ok("),
 ("X2 exclusive create: the commit replaces", "engram_plat.c",
  "        if (link(tmp, path) != 0) {", "        if (rename(tmp, path) != 0) {"),
 ("X3 keyfile: existence decided by a check that cannot fail", "engram_seal.c",
  "    rc = engram_file_size(path, &size);\n    if (rc == ENGRAM_OK) return ENGRAM_E_EXISTS;\n    if (rc != ENGRAM_E_NOTFOUND) return rc;",
  "    (void)size; if (engram_file_exists(path)) return ENGRAM_E_EXISTS;",
  "    rc = engram_file_create_atomic(path, buf, bn);", "    rc = engram_file_write_atomic(path, buf, bn);"),
 ("X1 atomic write: an allocation after the commit point", "engram_plat.c",
  "    dir = engram_dirname_dup(path);\n    if (!dir) { rc = ENGRAM_E_MEM; goto done; }\n\n    /* 1.",
  "\n    /* 1.",
  "    if (engram_io_tick()) { rc = ENGRAM_E_IO; goto done; }\n    do { dfd = open(dir",
  "    dir = engram_dirname_dup(path);\n    if (!dir) { rc = ENGRAM_E_MEM; goto done; }\n    if (engram_io_tick()) { rc = ENGRAM_E_IO; goto done; }\n    do { dfd = open(dir"),
]

if __name__ == "__main__":
    mutants.run(M, ["test_crypto", "test_persist"], by_file={"engram_plat.c": ["test_core"]},
                wine_only={"X3": "test_persist"})
