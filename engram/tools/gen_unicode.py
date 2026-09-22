#!/usr/bin/env python3
"""
gen_unicode.py -- generate ENGRAM's Unicode tables from the Unicode Character Database.

Emits:
  src/engram_unicode_tables.h     the tables the encoder reads (O(1) class lookup, packed folding)
  test/data/unicode_vectors.tsv   the EXPECTED class and folds of every covered codepoint, which
                                  test_text.c checks exhaustively against the C lookup

WHY GENERATED. Hand-typed Unicode tables are wrong in ways nobody notices for years. These come from
Python's unicodedata, whose version is written into the output, so a reader knows exactly which
edition of the standard the encoder implements.

WHY THE TEST VECTORS ARE A SEPARATE FILE. The C tables are data derived from this script; testing the
C lookup against this script's expectations tests the PACKING AND LOOKUP CODE end to end, over every
codepoint, rather than a hand-picked sample. The spot checks in test_text.c (e -> e-acute, sharp s ->
ss, ...) are the independent check on the DATA, and they come from the Unicode standard, not from here.

THE CHARACTER CLASSES
  WORD       letters and digits; a run of them is a word
  SPACE      separators and controls; collapsed to one space in the stream
  PUNCT      punctuation and symbols; breaks a word, but stays in the character stream
  MARK       a combining mark that is PART OF ITS LETTER -- Indic vowel signs and viramas, and every
             mark not listed below. Never removed: removing them destroys the word.
  IDEO       Han ideographs, Hiragana, Katakana: scripts written without spaces, so each character
             is a word of its own (the CJK practice of Lucene's CJK analyser)
  IGNORE     deleted outright: apostrophes (so "don't" == "dont"), zero-width joiners, variation
             selectors, format characters
  DIACRITIC  a combining mark that is DECORATION on its letter: the Latin/Greek/Cyrillic diacritic
             blocks, Hebrew niqqud and Arabic harakat. Dropped when compatibility folding is on (search
             engines strip these so text written with and without them matches), kept otherwise.

The distinction between MARK and DIACRITIC is the whole reason this table exists. The common shortcut
-- strip every combining mark to fold accents -- deletes Hindi vowel signs and the virama, and a
Hindi word with its vowels removed is not the same word.

Usage: gen_unicode.py <engram-root>
"""
import os
import sys
import unicodedata as U

CC = {"WORD": 0, "SPACE": 1, "PUNCT": 2, "MARK": 3, "IDEO": 4, "IGNORE": 5, "DIACRITIC": 6}

FOLD_RANGES = [
    (0x00A0, 0x024F),   # Latin-1 Supplement, Latin Extended-A, Latin Extended-B
    (0x0370, 0x052F),   # Greek and Coptic, Cyrillic, Cyrillic Supplement
    (0x0530, 0x058F),   # Armenian
    (0x1E00, 0x1FFF),   # Latin Extended Additional (Vietnamese), Greek Extended (polytonic)
    (0x2070, 0x209F),   # superscripts and subscripts
    (0x2460, 0x24FF),   # enclosed alphanumerics
    (0xFB00, 0xFB06),   # Latin ligatures
    (0xFF00, 0xFFEF),   # halfwidth and fullwidth forms
]

DIACRITIC_RANGES = [
    (0x0300, 0x036F), (0x1AB0, 0x1AFF), (0x1DC0, 0x1DFF), (0x20D0, 0x20FF), (0xFE20, 0xFE2F),
    (0x0591, 0x05C7),   # Hebrew points (only the Mn among them; punctuation there stays PUNCT)
    (0x064B, 0x065F), (0x0670, 0x0670),   # Arabic harakat
]

IDEO_RANGES = [
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
    (0x3041, 0x3096), (0x30A1, 0x30FA), (0x30FC, 0x30FC), (0xFF66, 0xFF9D),
]

APOSTROPHES = {0x0027, 0x2019, 0x02BC}
MAX_EXPANSION = 4


def in_ranges(c, ranges):
    return any(a <= c <= b for a, b in ranges)


def is_diacritic(c):
    return in_ranges(c, DIACRITIC_RANGES) and U.category(chr(c)).startswith("M")


def klass(c):
    if 0xD800 <= c <= 0xDFFF:
        return CC["PUNCT"]                       # surrogates never survive UTF-8 validation
    cat = U.category(chr(c))
    if c in APOSTROPHES:
        return CC["IGNORE"]
    if c == 0x200B:                              # ZERO WIDTH SPACE marks a word boundary
        return CC["SPACE"]
    if 0xFE00 <= c <= 0xFE0F:                    # variation selectors
        return CC["IGNORE"]
    if cat == "Cf":
        return CC["IGNORE"]
    if cat.startswith("Z") or cat == "Cc":
        return CC["SPACE"]
    if in_ranges(c, IDEO_RANGES):
        return CC["IDEO"]
    if cat.startswith("M"):
        return CC["DIACRITIC"] if is_diacritic(c) else CC["MARK"]
    if cat[0] in "LN" or cat in ("Co", "Cn"):
        return CC["WORD"]                        # unassigned: most future assignments are letters
    return CC["PUNCT"]


def strip_diacritics(s):
    return "".join(ch for ch in s if not is_diacritic(ord(ch)))


def fold_compat(ch):
    s = ch
    for _ in range(6):
        t = strip_diacritics(U.normalize("NFKD", s)).casefold()
        t = strip_diacritics(U.normalize("NFKD", t))
        if t == s:
            break
        s = t
    return s


def fold_case(ch):
    return ch.casefold()


def pack(entries):
    """Each fold becomes one uint32: bits 0-15 the target (length 1) or a pool offset (length > 1),
    bits 16-18 the length. Returns (words, pool)."""
    pool, words, seen = [], [], {}
    for s in entries:
        cps = [ord(x) for x in s]
        if len(cps) > MAX_EXPANSION:
            raise SystemExit("expansion longer than %d: %r" % (MAX_EXPANSION, s))
        if any(x > 0xFFFF for x in cps):
            raise SystemExit("non-BMP fold output: %r" % s)
        if len(cps) == 1:
            words.append((1 << 16) | cps[0])
            continue
        key = tuple(cps)
        if key not in seen:
            seen[key] = len(pool)
            pool.extend(cps)
        if seen[key] > 0xFFFF:
            raise SystemExit("pool overflow")
        words.append((len(cps) << 16) | seen[key])
    return words, pool


def c_array(name, ctype, values, per_line=12, fmt="0x%04X"):
    lines = ["static const %s %s[%d] = {" % (ctype, name, len(values))]
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(fmt % v for v in values[i:i + per_line]) + ",")
    lines.append("};")
    return "\n".join(lines)


def main(root):
    out_h = os.path.join(root, "src", "engram_unicode_tables.h")
    out_v = os.path.join(root, "test", "data", "unicode_vectors.tsv")

    # ---- class table, two-level ----------------------------------------------------------------
    blocks, index = {}, []
    for hi in range(256):
        blk = bytes(klass(hi * 256 + lo) for lo in range(256))
        index.append(blocks.setdefault(blk, len(blocks)))
    ordered = sorted(blocks.items(), key=lambda kv: kv[1])
    packed_blocks = []
    for blk, _ in ordered:
        for i in range(0, 256, 2):
            packed_blocks.append(blk[i] | (blk[i + 1] << 4))

    # ---- fold tables ----------------------------------------------------------------------------
    compat_all, case_all, range_rows, offset = [], [], [], 0
    for lo, hi in FOLD_RANGES:
        for c in range(lo, hi + 1):
            ch = chr(c)
            if U.category(ch) == "Cn":
                compat_all.append(ch)
                case_all.append(ch)
            else:
                compat_all.append(fold_compat(ch))
                case_all.append(fold_case(ch))
        range_rows.append((lo, hi, offset))
        offset += hi - lo + 1
    compat_words, compat_pool = pack(compat_all)
    case_words, case_pool = pack(case_all)

    ver = U.unidata_version
    h = []
    h.append("/* ==================================================================================================")
    h.append(" * engram_unicode_tables.h -- GENERATED by tools/gen_unicode.py from Unicode %s. DO NOT EDIT." % ver)
    h.append(" * ==================================================================================================")
    h.append(" * Regenerate with `python3 tools/gen_unicode.py .`; the reasoning behind every class is in that script.")
    h.append(" * Included by exactly one translation unit (engram_text.c).")
    h.append(" * ============================================================================================== */")
    h.append("#ifndef ENGRAM_UNICODE_TABLES_H")
    h.append("#define ENGRAM_UNICODE_TABLES_H")
    h.append("")
    h.append('#define ENGRAM_UNICODE_VERSION "%s"' % ver)
    h.append("#define ENGRAM_FOLD_MAX %d" % MAX_EXPANSION)
    h.append("")
    for k, v in sorted(CC.items(), key=lambda kv: kv[1]):
        h.append("#define ENGRAM_CC_%-10s %d" % (k, v))
    h.append("")
    h.append("/* Two-level class table for the BMP: ENGRAM_CC_BLOCK[ENGRAM_CC_INDEX[cp >> 8]] holds 256 classes,")
    h.append(" * two per byte, low nibble first. %d distinct blocks. */" % len(blocks))
    h.append(c_array("ENGRAM_CC_INDEX", "uint8_t", index, 16, "%d"))
    h.append("")
    h.append("#define ENGRAM_CC_NBLOCKS %d" % len(blocks))
    h.append(c_array("ENGRAM_CC_BLOCK", "uint8_t", packed_blocks, 16, "0x%02X"))
    h.append("")
    h.append("/* Fold ranges: codepoints [lo, hi] map through entries starting at `base`. */")
    h.append("typedef struct { uint16_t lo, hi, base; } engram_fold_range;")
    h.append("static const engram_fold_range ENGRAM_FOLD_RANGES[%d] = {" % len(range_rows))
    for lo, hi, base in range_rows:
        h.append("    { 0x%04X, 0x%04X, %d }," % (lo, hi, base))
    h.append("};")
    h.append("")
    h.append("/* Each entry: bits 0-15 = target codepoint (length 1) or offset into the pool (length > 1);")
    h.append(" * bits 16-18 = length. COMPAT = NFKD, diacritics stripped, case-folded. CASE = case-folded only. */")
    h.append(c_array("ENGRAM_FOLDTAB_COMPAT", "uint32_t", compat_words, 8, "0x%05X"))
    h.append(c_array("ENGRAM_FOLDTAB_COMPAT_POOL", "uint16_t", compat_pool or [0], 12))
    h.append(c_array("ENGRAM_FOLDTAB_CASE", "uint32_t", case_words, 8, "0x%05X"))
    h.append(c_array("ENGRAM_FOLDTAB_CASE_POOL", "uint16_t", case_pool or [0], 12))
    h.append("")
    h.append("#endif /* ENGRAM_UNICODE_TABLES_H */")
    with open(out_h, "w", newline="\n") as f:
        f.write("\n".join(h) + "\n")

    # ---- exhaustive test vectors, compactly --------------------------------------------------------
    # Classes as runs; folds only where they are NOT the identity. The C test checks every BMP
    # codepoint against the runs, and every codepoint in a fold range against the listed fold OR the
    # identity when unlisted -- so coverage is exhaustive while the file stays small.
    os.makedirs(os.path.dirname(out_v), exist_ok=True)
    with open(out_v, "w", newline="\n") as f:
        f.write("# GENERATED by tools/gen_unicode.py from Unicode %s. DO NOT EDIT.\n" % ver)
        f.write("# C <first> <last> <class>        class of every BMP codepoint (surrogates excluded)\n")
        f.write("# F <cp> <compat> <case>          every fold that is NOT the identity, in the fold ranges\n")
        prev, start_cp = None, 0
        for c in range(0x10000):
            k = None if 0xD800 <= c <= 0xDFFF else klass(c)
            if k != prev:
                if prev is not None:
                    f.write("C %04X %04X %d\n" % (start_cp, c - 1, prev))
                prev, start_cp = k, c
        if prev is not None:
            f.write("C %04X %04X %d\n" % (start_cp, 0xFFFF, prev))
        for lo, hi in FOLD_RANGES:
            for c in range(lo, hi + 1):
                ch = chr(c)
                if U.category(ch) == "Cn":
                    continue
                comp, case = fold_compat(ch), fold_case(ch)
                if comp == ch and case == ch:
                    continue
                f.write("F %04X %s %s\n" % (c, ",".join("%04X" % ord(x) for x in comp),
                                              ",".join("%04X" % ord(x) for x in case)))

    print("unicode %s: %d class blocks, %d fold entries, compat pool %d, case pool %d"
          % (ver, len(blocks), len(compat_words), len(compat_pool), len(case_pool)))
    print("wrote", out_h, "and", out_v)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
