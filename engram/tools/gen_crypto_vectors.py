#!/usr/bin/env python3
"""
gen_crypto_vectors.py -- freeze the test vectors engram_crypto.c is checked against.

TWO KINDS OF EVIDENCE, NEITHER TYPED BY HAND.
  1. The standards' own vectors, PARSED out of the RFC texts: RFC 8439 (ChaCha20, Poly1305 and the
     AEAD, including the eleven Poly1305 vectors that exercise the modular reduction's corners),
     RFC 4231 (HMAC-SHA-256), RFC 5869 (HKDF-SHA-256), RFC 7693 (BLAKE2b and its self-test grand hash)
     and RFC 9106 (Argon2id). Each RFC file is pinned by SHA-256 below; a different file is refused.
  2. Vectors from INDEPENDENT implementations at every boundary length the code has a branch for:
     Python's hashlib and hmac (SHA-256, HMAC, BLAKE2b; HKDF composed from hmac per RFC 5869),
     pyca/cryptography (ChaCha20-Poly1305, OpenSSL underneath) and argon2-cffi (the Argon2 reference
     implementation). An implementation that agrees with the RFC on one vector and with an independent
     implementation on a hundred boundary cases has little room left to be wrong.

Inputs are derived from SHA-256 in counter mode, so the same generator always produces the same file.

Usage:  gen_crypto_vectors.py <dir with rfc4231.txt rfc5869.txt rfc7693.txt rfc8439.txt rfc9106.txt> <out.h>
        (writes <out.h>, and seal_vectors.h beside it: the ENGRAM container and keyfile, built by an
        independent implementation of engram_seal.h's format)
        (needs: python3 with cryptography and argon2-cffi -- e.g. in a venv)
"""
import hashlib
import hmac
import os
import re
import struct
import sys

PINS = {
    "rfc4231.txt": "72178527ce93500e",
    "rfc5869.txt": None,
    "rfc7693.txt": "c943754888364fe2",
    "rfc8439.txt": "25bef70fbf7a07ff",
    "rfc9106.txt": "855c06f060379e34",
}


def load(d, name):
    raw = open(os.path.join(d, name), "rb").read()
    h = hashlib.sha256(raw).hexdigest()
    if PINS.get(name) and not h.startswith(PINS[name]):
        raise SystemExit("%s: sha256 %s is not the pinned file" % (name, h))
    text = raw.decode("ascii", "replace")
    # drop page furniture so a hex dump that crosses a page break reads as one block
    lines = [l for l in text.split("\n")
             if not re.match(r"^RFC \d+ ", l) and not re.search(r"\[Page \d+\]\s*$", l) and "\f" not in l]
    return lines, h


def hexdump_after(lines, i):
    """the 'NNN  xx xx ...' hex dump that starts at or after line i; offsets must run consecutively"""
    out = bytearray()
    while i < len(lines) and not re.match(r"^\s*\d{3}\s+[0-9a-f]{2}(\s|$)", lines[i]):
        i += 1
    while i < len(lines):
        l = lines[i]
        m = re.match(r"^\s*(\d{3})\s+((?:[0-9a-f]{2}\s?){1,16})", l)
        if not m:
            if l.strip() == "":
                i += 1
                continue
            break
        if int(m.group(1)) != len(out):
            break
        out += bytes.fromhex(m.group(2).replace(" ", ""))
        i += 1
    return bytes(out), i


def colonhex_after(lines, i):
    """'1a:e1:0b:...' on the lines after i (RFC 8439 2.8.2 prints its tag this way)"""
    out = bytearray()
    while i < len(lines):
        l = lines[i].strip()
        if re.match(r"^(?:[0-9a-f]{2}:)*[0-9a-f]{2}:?$", l):
            out += bytes.fromhex(l.replace(":", ""))
        elif l != "" or out:
            break
        i += 1
    return bytes(out), i


def find(lines, pat, start=0, end=None, must=True):
    for i in range(start, len(lines) if end is None else end):
        if re.search(pat, lines[i]):
            return i
    if must:
        raise SystemExit("pattern not found: " + pat)
    return -1


def plainhex_after(lines, i):
    """rows of bare 'XX XX ...' hex (RFC 8439 A.3 #5-#11), blank lines between rows allowed"""
    out = bytearray()
    while i < len(lines):
        l = lines[i]
        if re.match(r"^\s*(?:[0-9A-Fa-f]{2} )*[0-9A-Fa-f]{2}\s*$", l):
            out += bytes.fromhex(l.replace(" ", "").strip())
        elif l.strip() != "":
            break
        i += 1
    return bytes(out), i


def rfc8439(d):
    lines, h = load(d, "rfc8439.txt")
    v = {"block": [], "enc": [], "poly": [], "keygen": [], "aead": []}
    a1 = find(lines, r"^A\.1\.  The ChaCha20 Block Functions")
    a2 = find(lines, r"^A\.2\.  ChaCha20 Encryption")
    a3 = find(lines, r"^A\.3\.  Poly1305 Message Authentication Code")
    a4 = find(lines, r"^A\.4\.  Poly1305 Key Generation Using ChaCha20")
    a5 = find(lines, r"^A\.5\.  ChaCha20-Poly1305 AEAD Decryption")
    end = find(lines, r"^Appendix B|^Acknowledgements", a5)

    def vectors(lo, hi):
        starts = [i for i in range(lo, hi) if re.match(r"^\s*Test Vector #\d+:", lines[i])]
        return [(s, starts[k + 1] if k + 1 < len(starts) else hi) for k, s in enumerate(starts)]

    for s, e in vectors(a1, a2):
        key, _ = hexdump_after(lines, find(lines, r"^\s*Key:", s))
        nonce, _ = hexdump_after(lines, find(lines, r"^\s*Nonce:", s))
        ctr = int(re.search(r"Block Counter = (\d+)", "\n".join(lines[s:e])).group(1))
        ks, _ = hexdump_after(lines, find(lines, r"^\s*Keystream:", s))
        v["block"].append((key, nonce, ctr, ks))
    for s, e in vectors(a2, a3):
        key, _ = hexdump_after(lines, find(lines, r"^\s*Key:", s))
        nonce, _ = hexdump_after(lines, find(lines, r"^\s*Nonce:", s))
        ctr = int(re.search(r"Initial Block Counter = (\d+)", "\n".join(lines[s:e])).group(1))
        pt, _ = hexdump_after(lines, find(lines, r"^\s*Plaintext:", s))
        ct, _ = hexdump_after(lines, find(lines, r"^\s*Ciphertext:", s))
        v["enc"].append((key, nonce, ctr, pt, ct))
    for s, e in vectors(a3, a4):
        k = find(lines, r"^\s*One-time Poly1305 Key:", s, e, must=False)
        if k >= 0:
            key, _ = hexdump_after(lines, k)
            msg, _ = hexdump_after(lines, find(lines, r"^\s*Text to MAC:", s, e))
            tag, _ = hexdump_after(lines, find(lines, r"^\s*Tag:", s, e))
        else:                                     # #5-#11: R, S, data, tag as bare hex rows
            r, _ = plainhex_after(lines, find(lines, r"^\s*R:\s*$", s, e) + 1)
            sk, _ = plainhex_after(lines, find(lines, r"^\s*S:\s*$", s, e) + 1)
            msg, _ = plainhex_after(lines, find(lines, r"^\s*data:\s*$", s, e) + 1)
            tag, _ = plainhex_after(lines, find(lines, r"^\s*tag:\s*$", s, e) + 1)
            key = r + sk
        if len(key) != 32 or len(tag) != 16:
            raise SystemExit("RFC 8439 A.3 vector at line %d: key %d tag %d bytes" % (s, len(key), len(tag)))
        v["poly"].append((key, msg, tag))
    for s, e in vectors(a4, a5):
        key, _ = hexdump_after(lines, find(lines, r"^\s*The ChaCha20 Key", s))
        nonce, _ = hexdump_after(lines, find(lines, r"^\s*The nonce:", s))
        otk, _ = hexdump_after(lines, find(lines, r"^\s*Poly1305 one-time key:", s))
        v["keygen"].append((key, nonce, otk))
    # A.5: decryption
    key, _ = hexdump_after(lines, find(lines, r"^\s*The ChaCha20 Key", a5))
    ct, _ = hexdump_after(lines, find(lines, r"^\s*Ciphertext:", a5))
    nonce, _ = hexdump_after(lines, find(lines, r"^\s*The nonce:", a5))
    aad, _ = hexdump_after(lines, find(lines, r"^\s*The AAD:", a5))
    tag, _ = hexdump_after(lines, find(lines, r"^\s*Received Tag:", a5))
    pt, _ = hexdump_after(lines, find(lines, r"^\s*Plaintext::?", find(lines, r"^\s*Received Tag:", a5)))
    v["aead"].append((key, nonce, aad, pt, ct, tag))
    # 2.8.2: encryption
    s = find(lines, r"^2\.8\.2\.  Example and Test Vector for AEAD_CHACHA20_POLY1305")
    pt, _ = hexdump_after(lines, find(lines, r"^\s*Plaintext:", s))
    aad, _ = hexdump_after(lines, find(lines, r"^\s*AAD:", s))
    key, _ = hexdump_after(lines, find(lines, r"^\s*Key:", s))
    iv, _ = hexdump_after(lines, find(lines, r"^\s*IV:", s))
    fixed, _ = hexdump_after(lines, find(lines, r"^\s*32-bit fixed-common part:", s))
    ct, _ = hexdump_after(lines, find(lines, r"^\s*Ciphertext:", s))
    tag, _ = colonhex_after(lines, find(lines, r"^\s*Tag:", s) + 1)
    v["aead"].append((key, fixed + iv, aad, pt, ct, tag))
    # every vector the right shape, or nothing is written
    for key, nonce, ctr, ks in v["block"]:
        assert len(key) == 32 and len(nonce) == 12 and len(ks) == 64, "A.1"
    for key, nonce, ctr, pt, ct in v["enc"]:
        assert len(key) == 32 and len(nonce) == 12 and len(pt) == len(ct) > 0, "A.2"
    for key, nonce, otk in v["keygen"]:
        assert len(key) == 32 and len(nonce) == 12 and len(otk) == 32, "A.4"
    for key, nonce, aad, pt, ct, tag in v["aead"]:
        assert len(key) == 32 and len(nonce) == 12 and len(tag) == 16 and len(pt) == len(ct) > 0, "AEAD"
    assert len(v["block"]) == 5 and len(v["enc"]) == 3 and len(v["poly"]) == 11 and len(v["keygen"]) == 3
    return v, h


def rfc4231(d):
    lines, h = load(d, "rfc4231.txt")
    text = "\n".join(lines)
    out = []
    for n in range(1, 8):
        # anchored to a whole line: the table of contents holds the same titles, followed by dots
        s = re.search(r"^4\.%d\.  Test Case %d\s*$" % (n + 1, n), text, re.M).start()
        e = (re.search(r"^4\.%d\.  Test Case %d\s*$" % (n + 2, n + 1), text, re.M) if n < 7 else
             re.search(r"^5\.  Security Considerations\s*$", text, re.M)).start()
        body = text[s:e]

        def field(name):
            ls = body.split("\n")
            for k, l in enumerate(ls):
                # "=" optional: RFC 4231 test case 3 prints "Key" without it
                m = re.match(r"^\s*" + re.escape(name) + r"\s*=?\s+([0-9a-f]+)", l)
                if not m:
                    continue
                val = m.group(1)
                for l2 in ls[k + 1:]:                 # continuation: indented hex, no "name ="
                    mm = re.match(r"^\s{10,}([0-9a-f]+)(\s|$)", l2)
                    if mm and "=" not in l2:
                        val += mm.group(1)
                    else:
                        break
                return bytes.fromhex(val)
            raise SystemExit("RFC 4231 test case %d: no %s" % (n, name))
        key, data, mac = field("Key"), field("Data"), field("HMAC-SHA-256")
        out.append((key, data, mac))
    return out, h


def rfc5869(d):
    lines, h = load(d, "rfc5869.txt")
    text = "\n".join(lines)
    out = []
    for n in (1, 2, 3):
        s = re.search(r"^A\.%d\.  Test Case %d\s*$" % (n, n), text, re.M).start()
        e = re.search(r"^A\.%d\.  Test Case %d\s*$" % (n + 1, n + 1), text, re.M).start()
        body = text[s:e]

        def field(name):
            m = re.search(r"^\s*" + name + r"\s*=\s*(.*)$", body, re.M)
            val, rest = "", body[m.start(1):].split("\n")
            for k, l in enumerate(rest):
                if k > 0 and re.match(r"^\s*[A-Za-z]+\s*=", l):
                    break
                val += re.sub(r"\(\d+ octets\)", "", l).replace("0x", "").strip()
            return val
        ikm, salt, info = (bytes.fromhex(field(x)) for x in ("IKM", "salt", "info"))
        L = int(field("L"))
        prk, okm = bytes.fromhex(field("PRK")), bytes.fromhex(field("OKM"))
        out.append((ikm, salt, info, L, prk, okm))
    return out, h


def rfc7693(d):
    lines, h = load(d, "rfc7693.txt")
    text = "\n".join(lines)
    m = re.search(r'BLAKE2b-512\("abc"\) = ((?:[0-9A-F]{2} ?)+(?:\n\s+(?:[0-9A-F]{2} ?)+)*)', text)
    abc = bytes.fromhex(re.sub(r"\s+", "", m.group(1)))
    m = re.search(r"const uint8_t blake2b_res\[32\] = \{([^}]*)\}", text)
    res = bytes(int(x, 16) for x in re.findall(r"0x([0-9A-F]{2})", m.group(1)))
    return abc, res, h


def rfc9106(d):
    lines, h = load(d, "rfc9106.txt")
    text = "\n".join(lines)
    s = re.search(r"^5\.3\.  Argon2id Test Vectors\s*$", text, re.M).start()
    m = re.search(r"Tag: ((?:[0-9a-f]{2} ?)+(?:\n\s+(?:[0-9a-f]{2} ?)+)*)", text[s:])
    tag = bytes.fromhex(re.sub(r"\s+", "", m.group(1)))
    return tag, h


def stream(label, n):
    out, c = b"", 0
    while len(out) < n:
        out += hashlib.sha256(("%s/%d" % (label, c)).encode()).digest()
        c += 1
    return out[:n]


def hkdf_ref(salt, ikm, info, L):
    prk = hmac.new(salt if salt else b"\0" * 32, ikm, hashlib.sha256).digest()
    t, okm, i = b"", b"", 1
    while len(okm) < L:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return prk, okm[:L]


def c_bytes(b):
    if not b:
        return '""'
    s = "".join("\\x%02x" % x for x in b)
    return "\n        ".join('"' + s[i:i + 96] + '"' for i in range(0, len(s), 96))


def main(rfcdir, out):
    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    import argon2.low_level as a2
    v8439, h8439 = rfc8439(rfcdir)
    v4231, h4231 = rfc4231(rfcdir)
    v5869, h5869 = rfc5869(rfcdir)
    abc, b2res, h7693 = rfc7693(rfcdir)
    a2tag, h9106 = rfc9106(rfcdir)
    L = []
    w = L.append
    w("/* GENERATED by tools/gen_crypto_vectors.py -- DO NOT EDIT.")
    w(" * RFC texts (SHA-256): 4231 %s, 5869 %s, 7693 %s, 8439 %s, 9106 %s." % (h4231[:16], h5869[:16], h7693[:16], h8439[:16], h9106[:16]))
    w(" * Cross-check vectors: Python hashlib/hmac, pyca/cryptography, argon2-cffi. */")
    w("#ifndef ENGRAM_CRYPTO_VECTORS_H")
    w("#define ENGRAM_CRYPTO_VECTORS_H")
    cv_t = ["/* byte arrays, not string literals: C99 promises string literals only up to 4095 characters */",
            "#ifndef ENGRAM_CV_T",
            "#define ENGRAM_CV_T",
            "typedef struct { const uint8_t *a, *b, *c, *d, *e, *f; size_t na, nb, nc, nd, ne, nf; unsigned x, y, z, u; } cv_t;",
            "#endif"]
    L.extend(cv_t)

    def table(name, rows):
        refs = []
        for ri, r in enumerate(rows):
            strs = list(r[0]) + [b""] * (6 - len(r[0]))
            names = []
            for fi, x in enumerate(strs):
                if not x:
                    names.append("NULL")
                    continue
                an = "%s_%d%s" % (name, ri, "abcdef"[fi])
                hx = ",".join("0x%02x" % c for c in x)
                w("static const uint8_t %s[%d] = {" % (an, len(x)))
                for k in range(0, len(hx), 100):
                    w("    " + hx[k:k + 100])
                w("};")
                names.append(an)
            refs.append((names, strs, list(r[1]) + [0] * (4 - len(r[1]))))
        w("static const cv_t %s[] = {" % name)
        for names, strs, ints in refs:
            w("    { " + ", ".join(names) + ", " + ", ".join("%du" % len(x) for x in strs) + ", " +
              ", ".join("%du" % x for x in ints) + " },")
        w("};")
        w("#define %s_N (sizeof %s / sizeof %s[0])" % (name.upper(), name, name))

    # ---- the RFC vectors ----
    table("rfc8439_block", [((k, n, ks), (c,)) for k, n, c, ks in v8439["block"]])
    table("rfc8439_enc", [((k, n, p, ct), (c,)) for k, n, c, p, ct in v8439["enc"]])
    table("rfc8439_poly", [((k, m, t), ()) for k, m, t in v8439["poly"]])
    table("rfc8439_keygen", [((k, n, o), ()) for k, n, o in v8439["keygen"]])
    table("rfc8439_aead", [((k, n, a, p, c, t), ()) for k, n, a, p, c, t in v8439["aead"]])
    table("rfc4231_hmac", [((k, d, m), ()) for k, d, m in v4231])
    table("rfc5869_hkdf", [((ikm, salt, info, prk, okm), (Lx,)) for ikm, salt, info, Lx, prk, okm in v5869])
    table("rfc7693_blake2b", [((b"abc", abc, b2res), ())])
    table("rfc9106_argon2id", [((b"\x01" * 32, b"\x02" * 16, b"\x03" * 8, b"\x04" * 12, a2tag), (3, 32, 4, 32))])

    # ---- independent implementations, at every boundary ----
    lens = [0, 1, 3, 31, 32, 33, 55, 56, 57, 63, 64, 65, 111, 112, 119, 120, 127, 128, 129, 191, 192, 255, 256, 1000]
    rows = []
    for n in lens:
        m = stream("sha/%d" % n, n)
        rows.append(((m, hashlib.sha256(m).digest()), ()))
    table("x_sha256", rows)
    rows = []
    for kl in [0, 1, 20, 31, 32, 33, 63, 64, 65, 100, 131]:
        for n in [0, 1, 64, 65, 200]:
            k, m = stream("hk/%d" % kl, kl), stream("hm/%d/%d" % (kl, n), n)
            rows.append(((k, m, hmac.new(k, m, hashlib.sha256).digest()), ()))
    table("x_hmac", rows)
    rows = []
    for sl, il, Lx in [(0, 0, 1), (0, 10, 32), (13, 0, 33), (32, 5, 64), (80, 20, 100), (1, 1, 255), (16, 16, 1000), (64, 64, 8160)]:
        salt, ikm, info = stream("hs/%d" % sl, sl), stream("hi/%d" % Lx, 22 + sl), stream("hn/%d" % il, il)
        prk, okm = hkdf_ref(salt, ikm, info, Lx)
        rows.append(((ikm, salt, info, prk, okm), (Lx,)))
    table("x_hkdf", rows)
    rows = []
    for ol in [1, 16, 20, 32, 48, 63, 64]:
        for kl in [0, 1, 32, 63, 64]:
            for n in [0, 1, 127, 128, 129, 255, 256, 1000]:
                k, m = stream("bk/%d/%d" % (ol, kl), kl), stream("bm/%d/%d/%d" % (ol, kl, n), n)
                d = hashlib.blake2b(m, digest_size=ol, key=k).digest()
                rows.append(((k, m, d), (ol,)))
    table("x_blake2b", rows)
    rows = []
    for al in [0, 1, 12, 16, 17, 100]:
        for n in [0, 1, 15, 16, 17, 63, 64, 65, 200, 1000]:
            k, nonce = stream("ak/%d/%d" % (al, n), 32), stream("an/%d/%d" % (al, n), 12)
            a, p = stream("aa/%d" % al, al), stream("ap/%d/%d" % (al, n), n)
            ct = ChaCha20Poly1305(k).encrypt(nonce, p, a)
            rows.append(((k, nonce, a, p, ct[:-16], ct[-16:]), ()))
    table("x_aead", rows)
    rows = []
    for t, m, p, ol, pl, sl in [(1, 8, 1, 4, 0, 8), (1, 16, 2, 16, 5, 8), (2, 64, 1, 32, 32, 16), (3, 32, 4, 32, 12, 16),
                                (1, 64, 4, 64, 8, 32), (2, 100, 3, 100, 20, 11), (1, 1024, 1, 32, 16, 16),
                                (1, 65, 1, 1024, 7, 9), (4, 256, 2, 72, 64, 16), (1, 2048, 8, 33, 3, 8)]:
        pwd, salt = stream("pw/%d/%d" % (t, m), pl), stream("sa/%d/%d" % (t, m), sl)
        tag = a2.hash_secret_raw(pwd, salt, time_cost=t, memory_cost=m, parallelism=p, hash_len=ol, type=a2.Type.ID)
        rows.append(((pwd, salt, b"", b"", tag), (t, m, p, ol)))
    table("x_argon2id", rows)

    w("#endif")
    open(out, "w", newline="\n").write("\n".join(L) + "\n")
    print("wrote %s: %d lines" % (out, len(L)))

    # ---- the ENGRAM container and keyfile, re-implemented here from engram_seal.h's specification ----
    # A second header (seal_vectors.h beside the first), so a suite that does not use these tables
    # does not carry them -- an unused static table is a warning, and a warning is an error here.
    out = os.path.join(os.path.dirname(out) or ".", "seal_vectors.h")
    del L[:]
    w("/* GENERATED by tools/gen_crypto_vectors.py -- DO NOT EDIT.")
    w(" * The ENGRAM container (engram_seal.h) and keyfile, built by an independent Python implementation of")
    w(" * the format: hashlib, hmac, pyca/cryptography (ChaCha20-Poly1305, Argon2id with associated data). */")
    w("#ifndef ENGRAM_SEAL_VECTORS_H")
    w("#define ENGRAM_SEAL_VECTORS_H")
    L.extend(cv_t)
    # (pyca/cryptography for the AEAD and Argon2id -- it takes the associated data argon2-cffi cannot)
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

    def container(kind, key, salt, payload):
        hdr = b"ENGRAM\x1a\x0a" + struct.pack("<HHIQ", 1, kind, 1 if key else 0, len(payload)) + salt + b"\0" * 8
        assert len(hdr) == 64
        if key:
            _, okm = hkdf_ref(salt, key, b"ENGRAM seal v1" + struct.pack("<H", kind), 44)
            return hdr + ChaCha20Poly1305(okm[:32]).encrypt(okm[32:], payload, hdr) + b"\0" * 16
        return hdr + payload + hashlib.sha256(hdr + payload).digest()

    rows = []
    for kind in [1, 2, 5, 99, 0xFFFF]:
        for n in [0, 1, 15, 16, 17, 63, 64, 65, 200] + ([1000] if kind == 1 else []):
            for sealed in [0, 1]:
                key = stream("sk/%d/%d" % (kind, n), 32) if sealed else b""
                salt, pl = stream("ss/%d/%d/%d" % (kind, n, sealed), 32), stream("sp/%d/%d" % (kind, n), n)
                rows.append(((key, salt, pl, container(kind, key, salt, pl)), (kind, sealed)))
    table("x_seal", rows)
    rows = []
    for t, m, p, pl in [(1, 8, 1, 0), (1, 8, 1, 1), (2, 64, 1, 12), (1, 32, 2, 64), (3, 256, 1, 200), (1, 1024, 4, 33)]:
        pwd, salt = stream("kp/%d/%d/%d" % (t, m, pl), pl), stream("ks/%d/%d/%d" % (t, m, pl), 16)
        master = Argon2id(salt=salt, length=32, iterations=t, lanes=p, memory_cost=m,
                          ad=b"ENGRAM master key").derive(pwd)
        verifier = hmac.new(master, b"ENGRAM key check", hashlib.sha256).digest()
        payload = struct.pack("<IIII", 1, t, m, p) + salt + verifier
        rows.append(((pwd, salt, master, payload), (t, m, p)))
    table("x_keyfile", rows)
    w("#endif")
    open(out, "w", newline="\n").write("\n".join(L) + "\n")
    print("wrote %s: %d lines" % (out, len(L)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
