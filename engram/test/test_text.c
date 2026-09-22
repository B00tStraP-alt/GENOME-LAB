/* ==================================================================================================
 * test_text.c -- P1.2 (part 1): UTF-8, classes, folding and the normalised stream, proven.
 * ==================================================================================================
 * THREE KINDS OF EVIDENCE, AND WHY ALL THREE
 *   1. EXHAUSTIVE: every BMP codepoint's class and both of its folds are checked against
 *      test/data/unicode_vectors.tsv, generated from the Unicode Character Database. That proves the
 *      C lookup and the table packing, over ~65,000 codepoints, not a sample.
 *   2. INDEPENDENT: the malformed-UTF-8 expectations come from Python's decoder, and the folding spot
 *      checks come from the Unicode standard itself -- neither from the generator this code was built
 *      from. Checking the tables only against their own generator would prove consistency, not truth.
 *   3. BEHAVIOURAL: what a person experiences -- "Straße" finds "strasse", "don't" finds "dont", a
 *      Hindi word keeps its vowels.
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_plat.h"
#include "../src/engram_text.h"

#include <stdlib.h>
#include <string.h>

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

/* ==================================================================================================
 * UTF-8
 * ============================================================================================== */
typedef struct {
    const char *bytes;
    unsigned    ncp;
    uint32_t    cp[4];
    const char *why;
} utf8_case;

/* Expected sequences computed by Python's bytes.decode('utf-8', 'replace'), an independent
 * implementation of the Unicode maximal-subpart practice. */
static const utf8_case UTF8_CASES[] = {
    { "\xF0\x80\x80",             3, { 0xFFFD, 0xFFFD, 0xFFFD },         "overlong 4-byte prefix" },
    { "\xE2\x82",                 1, { 0xFFFD },                         "truncated euro sign" },
    { "\xED\xA0\x80",             3, { 0xFFFD, 0xFFFD, 0xFFFD },         "UTF-16 surrogate" },
    { "\xC0\xAF",                 2, { 0xFFFD, 0xFFFD },                 "overlong slash" },
    { "\xF4\x90\x80\x80",         4, { 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD }, "above U+10FFFF" },
    { "\x61\x80\x62",             3, { 0x0061, 0xFFFD, 0x0062 },         "lone continuation" },
    { "\xE2\x82\xAC\x78\xE2",     3, { 0x20AC, 0x0078, 0xFFFD },         "valid, valid, truncated" },
    { "\xF1\x80\x80\xE1\x80\xC2", 3, { 0xFFFD, 0xFFFD, 0xFFFD },         "WHATWG: 3 maximal subparts" },
    { "\xFF\xFE\xFD",             3, { 0xFFFD, 0xFFFD, 0xFFFD },         "never-valid bytes" },
};

static void test_utf8(void)
{
    uint32_t cp, got;
    uint8_t enc[4];
    size_t n, k, off, bad;
    int valid, ok;
    unsigned c;
    unsigned long mism = 0;

    ET_SECTION("utf8: every scalar value round-trips through encode and decode");
    for (cp = 0; cp <= 0x10FFFFu; cp++) {
        if (cp >= 0xD800u && cp <= 0xDFFFu) {
            if (engram_utf8_encode(cp, enc) != 0u) mism++;         /* surrogates must be refused */
            continue;
        }
        n = engram_utf8_encode(cp, enc);
        k = engram_utf8_decode(enc, n, &got, &valid);
        if (!n || k != n || !valid || got != cp) mism++;
    }
    ET_CHECKF(mism == 0u, "%lu scalar values failed to round-trip", mism);
    ET_EQ_U64(engram_utf8_encode(0x110000u, enc), 0u);

    ET_SECTION("utf8: encoded lengths are the shortest form");
    ET_EQ_U64(engram_utf8_encode(0x7Fu, enc), 1u);
    ET_EQ_U64(engram_utf8_encode(0x80u, enc), 2u);
    ET_EQ_U64(engram_utf8_encode(0x7FFu, enc), 2u);
    ET_EQ_U64(engram_utf8_encode(0x800u, enc), 3u);
    ET_EQ_U64(engram_utf8_encode(0xFFFFu, enc), 3u);
    ET_EQ_U64(engram_utf8_encode(0x10000u, enc), 4u);
    ET_EQ_U64(engram_utf8_encode(0x10FFFFu, enc), 4u);

    ET_SECTION("utf8: malformed input matches Python's maximal-subpart decoding exactly");
    for (c = 0; c < sizeof UTF8_CASES / sizeof UTF8_CASES[0]; c++) {
        const utf8_case *t = &UTF8_CASES[c];
        const uint8_t *p = (const uint8_t *)t->bytes;
        size_t len = strlen(t->bytes);
        unsigned count = 0;
        ok = 1;
        for (off = 0; off < len; ) {
            k = engram_utf8_decode(p + off, len - off, &got, &valid);
            if (k == 0u) { ok = 0; break; }
            if (count >= 4u || got != t->cp[count]) ok = 0;
            count++;
            off += k;
        }
        ET_CHECKF(ok && count == t->ncp, "%s: got %u codepoints, want %u", t->why, count, t->ncp);
    }

    ET_SECTION("utf8: validate reports the offset of the first bad byte");
    ET_OK(engram_utf8_validate((const uint8_t *)"plain ascii", 11u, &bad));
    ET_OK(engram_utf8_validate((const uint8_t *)"\xC3\xA9t\xC3\xA9", 5u, &bad));
    ET_RC(engram_utf8_validate((const uint8_t *)"ab\xC0\xAF", 4u, &bad), ENGRAM_E_UTF8);
    ET_EQ_U64(bad, 2u);
    ET_OK(engram_utf8_validate(NULL, 0u, &bad));
    ET_RC(engram_utf8_validate(NULL, 3u, &bad), ENGRAM_E_ARG);
}

/* ==================================================================================================
 * EXHAUSTIVE: every BMP codepoint against the Unicode Character Database
 * ============================================================================================== */
typedef struct {
    uint8_t  cls[0x10000];
    uint8_t  has_fold[0x10000];
    uint16_t compat[0x10000][4];
    uint8_t  ncompat[0x10000];
    uint16_t casef[0x10000][4];
    uint8_t  ncase[0x10000];
} ucd_expect;

static unsigned parse_seq(const char *s, uint16_t out[4])
{
    unsigned n = 0;
    char *end;
    while (*s && *s != ' ' && *s != '\n' && n < 4u) {
        out[n++] = (uint16_t)strtoul(s, &end, 16);
        if (end == s) break;                           /* not a number: stop, never loop */
        s = end;
        if (*s == ',') s++;
    }
    return n;
}

static void test_ucd_exhaustive(void)
{
    char *path;
    uint8_t *data = NULL;
    size_t len = 0;
    ucd_expect *E;
    const char *line;
    uint32_t cp, out[ENGRAM_FOLD_OUT_MAX];
    unsigned long class_bad = 0, compat_bad = 0, case_bad = 0, class_rows = 0, fold_rows = 0;
    unsigned shown = 0;

    ET_SECTION("ucd: load the generated vectors");
    path = engram_path_join(data_dir(), "unicode_vectors.tsv");
    ET_CHECK(path != NULL);
    ET_OK(engram_file_read(path, &data, &len));
    engram_free(path);
    E = (ucd_expect *)engram_calloc(1u, sizeof *E);
    ET_CHECK(E != NULL);
    if (!data || !E) { engram_free(data); engram_free(E); return; }
    memset(E->cls, 0xFF, sizeof E->cls);

    for (line = (const char *)data; line && *line; ) {
        const char *nl = strchr(line, '\n');
        if (line[0] == 'C' && line[1] == ' ') {
            char *q;
            unsigned long a = strtoul(line + 2, &q, 16), b = strtoul(q, &q, 16), k = strtoul(q, &q, 10);
            for (cp = (uint32_t)a; cp <= (uint32_t)b && cp < 0x10000u; cp++) E->cls[cp] = (uint8_t)k;
            class_rows++;
        } else if (line[0] == 'F' && line[1] == ' ') {
            char *q;
            unsigned long c = strtoul(line + 2, &q, 16);
            if (c < 0x10000u) {
                E->has_fold[c] = 1u;
                E->ncompat[c] = (uint8_t)parse_seq(q + 1, E->compat[c]);
                q = strchr(q + 1, ' ');
                if (q) E->ncase[c] = (uint8_t)parse_seq(q + 1, E->casef[c]);
            }
            fold_rows++;
        }
        line = nl ? nl + 1 : NULL;
    }
    printf("       Unicode %s: %lu class runs, %lu non-identity folds\n",
           engram_unicode_version(), class_rows, fold_rows);
    ET_CHECK(class_rows > 100u);
    ET_CHECK(fold_rows > 1000u);

    ET_SECTION("ucd: class of every BMP codepoint (surrogates excluded)");
    for (cp = 0; cp < 0x10000u; cp++) {
        if (cp >= 0xD800u && cp <= 0xDFFFu) continue;
        if (E->cls[cp] == 0xFFu || engram_cp_class(cp) != E->cls[cp]) {
            if (shown++ < 5u) printf("         class U+%04X: got %u want %u\n", cp, engram_cp_class(cp), E->cls[cp]);
            class_bad++;
        }
    }
    ET_CHECKF(class_bad == 0u, "%lu codepoints misclassified", class_bad);

    ET_SECTION("ucd: both folds of every BMP codepoint");
    for (cp = 0; cp < 0x10000u; cp++) {
        size_t n, i;
        uint32_t id = (cp >= 'A' && cp <= 'Z') ? cp + 32u : cp;
        if (cp >= 0xD800u && cp <= 0xDFFFu) continue;
        n = engram_cp_fold(cp, ENGRAM_FOLD_COMPAT, out);
        if (E->has_fold[cp]) {
            int m = (n == E->ncompat[cp]);
            for (i = 0; m && i < n; i++) if (out[i] != E->compat[cp][i]) m = 0;
            if (!m) compat_bad++;
        } else if (n != 1u || out[0] != id) compat_bad++;
        n = engram_cp_fold(cp, ENGRAM_FOLD_CASE, out);
        if (E->has_fold[cp]) {
            int m = (n == E->ncase[cp]);
            for (i = 0; m && i < n; i++) if (out[i] != E->casef[cp][i]) m = 0;
            if (!m) case_bad++;
        } else if (n != 1u || out[0] != id) case_bad++;
    }
    ET_CHECKF(compat_bad == 0u, "%lu compat folds wrong", compat_bad);
    ET_CHECKF(case_bad == 0u, "%lu case folds wrong", case_bad);

    ET_SECTION("ucd: FOLD_NONE is the identity on every BMP codepoint");
    for (compat_bad = 0, cp = 0; cp < 0x10000u; cp++)
        if (engram_cp_fold(cp, ENGRAM_FOLD_NONE, out) != 1u || out[0] != cp) compat_bad++;
    ET_EQ_U64(compat_bad, 0u);

    engram_free(E);
    engram_free(data);
}

/* ==================================================================================================
 * INDEPENDENT: facts from the Unicode standard, not from the generator
 * ============================================================================================== */
static int fold_is(uint32_t cp, engram_fold mode, const uint32_t *want, size_t nwant)
{
    uint32_t out[ENGRAM_FOLD_OUT_MAX];
    size_t n = engram_cp_fold(cp, mode, out), i;
    if (n != nwant) return 0;
    for (i = 0; i < n; i++) if (out[i] != want[i]) return 0;
    return 1;
}

#define FOLD1(cp, mode, a)       do { uint32_t w_[1] = { (a) };      ET_CHECKF(fold_is((cp), (mode), w_, 1u), "U+%04X", (unsigned)(cp)); } while (0)
#define FOLD2(cp, mode, a, b)    do { uint32_t w_[2] = { (a), (b) }; ET_CHECKF(fold_is((cp), (mode), w_, 2u), "U+%04X", (unsigned)(cp)); } while (0)

static void test_standard_facts(void)
{
    ET_SECTION("fold: facts from the Unicode standard");
    FOLD1(0x00C9, ENGRAM_FOLD_COMPAT, 'e');       /* E WITH ACUTE -> e            */
    FOLD1(0x00C9, ENGRAM_FOLD_CASE,   0x00E9);    /*               -> e with acute */
    FOLD2(0x00DF, ENGRAM_FOLD_COMPAT, 's', 's');  /* SHARP S -> ss (full folding)  */
    FOLD2(0x00DF, ENGRAM_FOLD_CASE,   's', 's');
    FOLD1(0x03A9, ENGRAM_FOLD_CASE,   0x03C9);    /* OMEGA -> omega                */
    FOLD1(0x03C2, ENGRAM_FOLD_CASE,   0x03C3);    /* final sigma -> sigma          */
    FOLD1(0x03AC, ENGRAM_FOLD_COMPAT, 0x03B1);    /* alpha with tonos -> alpha     */
    FOLD1(0x0401, ENGRAM_FOLD_COMPAT, 0x0435);    /* CYRILLIC IO -> ie             */
    FOLD1(0x0401, ENGRAM_FOLD_CASE,   0x0451);    /*              -> io            */
    FOLD1(0x0416, ENGRAM_FOLD_CASE,   0x0436);    /* ZHE -> zhe                    */
    FOLD2(0xFB01, ENGRAM_FOLD_COMPAT, 'f', 'i');  /* LIGATURE FI -> fi             */
    FOLD1(0xFF26, ENGRAM_FOLD_COMPAT, 'f');       /* FULLWIDTH F -> f              */
    FOLD1(0x1EC7, ENGRAM_FOLD_COMPAT, 'e');       /* Vietnamese e-circumflex-dot   */
    FOLD1(0x0130, ENGRAM_FOLD_COMPAT, 'i');       /* I WITH DOT ABOVE -> i         */
    FOLD1(0x2460, ENGRAM_FOLD_COMPAT, '1');       /* CIRCLED DIGIT ONE -> 1        */
    FOLD1(0x24B6, ENGRAM_FOLD_COMPAT, 'a');       /* CIRCLED CAPITAL A -> a        */
    FOLD1(0x00E9, ENGRAM_FOLD_NONE,   0x00E9);    /* NONE changes nothing          */

    ET_SECTION("fold: the documented gap -- scripts outside the tables pass through");
    FOLD1(0x10A0, ENGRAM_FOLD_CASE, 0x10A0);      /* GEORGIAN CAPITAL AN: not folded, stated */

    ET_SECTION("class: facts that decide what a word is");
    ET_EQ_U64(engram_cp_class('a'),      ENGRAM_CLASS_WORD);
    ET_EQ_U64(engram_cp_class('7'),      ENGRAM_CLASS_WORD);
    ET_EQ_U64(engram_cp_class(' '),      ENGRAM_CLASS_SPACE);
    ET_EQ_U64(engram_cp_class('\t'),     ENGRAM_CLASS_SPACE);
    ET_EQ_U64(engram_cp_class(','),      ENGRAM_CLASS_PUNCT);
    ET_EQ_U64(engram_cp_class('-'),      ENGRAM_CLASS_PUNCT);
    ET_EQ_U64(engram_cp_class(0x0027),   ENGRAM_CLASS_IGNORE);   /* apostrophe            */
    ET_EQ_U64(engram_cp_class(0x2019),   ENGRAM_CLASS_IGNORE);   /* right single quote    */
    ET_EQ_U64(engram_cp_class(0x200D),   ENGRAM_CLASS_IGNORE);   /* zero width joiner     */
    ET_EQ_U64(engram_cp_class(0xFE0F),   ENGRAM_CLASS_IGNORE);   /* variation selector-16 */
    ET_EQ_U64(engram_cp_class(0x200B),   ENGRAM_CLASS_SPACE);    /* zero width space      */
    ET_EQ_U64(engram_cp_class(0x00A0),   ENGRAM_CLASS_SPACE);    /* no-break space        */
    ET_EQ_U64(engram_cp_class(0x3000),   ENGRAM_CLASS_SPACE);    /* ideographic space     */
    ET_EQ_U64(engram_cp_class(0x4E2D),   ENGRAM_CLASS_IDEO);     /* han: middle           */
    ET_EQ_U64(engram_cp_class(0x3042),   ENGRAM_CLASS_IDEO);     /* hiragana a            */
    ET_EQ_U64(engram_cp_class(0x30AB),   ENGRAM_CLASS_IDEO);     /* katakana ka           */
    ET_EQ_U64(engram_cp_class(0xD55C),   ENGRAM_CLASS_WORD);     /* hangul: space-delimited */
    ET_EQ_U64(engram_cp_class(0x0301),   ENGRAM_CLASS_DIACRITIC);/* combining acute       */
    ET_EQ_U64(engram_cp_class(0x05B8),   ENGRAM_CLASS_DIACRITIC);/* hebrew qamats         */
    ET_EQ_U64(engram_cp_class(0x093F),   ENGRAM_CLASS_MARK);     /* devanagari vowel sign i -- KEPT */
    ET_EQ_U64(engram_cp_class(0x094D),   ENGRAM_CLASS_MARK);     /* devanagari virama        -- KEPT */
    ET_EQ_U64(engram_cp_class(0x1F600),  ENGRAM_CLASS_PUNCT);    /* emoji                 */
    ET_EQ_U64(engram_cp_class(0x20000),  ENGRAM_CLASS_IDEO);     /* CJK extension B       */
    ET_EQ_U64(engram_cp_class(0xE0001),  ENGRAM_CLASS_IGNORE);   /* language tag          */
    ET_EQ_U64(engram_cp_class(0x10000),  ENGRAM_CLASS_WORD);     /* linear B: a letter    */
    ET_EQ_U64(engram_cp_class(0x110000), ENGRAM_CLASS_PUNCT);    /* beyond Unicode        */
    ET_CHECK(strlen(engram_unicode_version()) >= 4u);
}

/* ==================================================================================================
 * BEHAVIOURAL: what a person experiences
 * ============================================================================================== */
static void norm_is(const char *in, engram_fold fold, const char *want)
{
    char *out = NULL;
    size_t len = 0;
    engram_rc rc = engram_text_normalize(in, strlen(in), fold, ENGRAM_UTF8_STRICT, &out, &len);
    ET_CHECKF(rc == ENGRAM_OK && out && strcmp(out, want) == 0 && len == strlen(want),
              "normalize(\"%s\") = \"%s\" (rc %s), want \"%s\"", in, out ? out : "(null)",
              engram_rcname(rc), want);
    engram_free(out);
}

static void test_normalize(void)
{
    char *out = NULL;
    size_t len = 0;
    engram_textit it;
    uint32_t cp;
    unsigned k, n;

    ET_SECTION("normalize: whitespace collapses and trims; case folds");
    norm_is("  Hello,   WORLD!  ", ENGRAM_FOLD_COMPAT, "hello, world!");
    norm_is("a\t\n\r b", ENGRAM_FOLD_NONE, "a b");
    norm_is("", ENGRAM_FOLD_COMPAT, "");
    norm_is("   ", ENGRAM_FOLD_COMPAT, "");

    ET_SECTION("normalize: accents fold under COMPAT, and are kept under CASE");
    norm_is("Caf\xC3\xA9 Cr\xC3\xA8me", ENGRAM_FOLD_COMPAT, "cafe creme");
    norm_is("Caf\xC3\xA9 Cr\xC3\xA8me", ENGRAM_FOLD_CASE, "caf\xC3\xA9 cr\xC3\xA8me");
    norm_is("Vi\xE1\xBB\x87t Nam", ENGRAM_FOLD_COMPAT, "viet nam");
    norm_is("Stra\xC3\x9F" "e", ENGRAM_FOLD_COMPAT, "strasse");
    norm_is("\xEF\xBC\xA1\xEF\xBC\xA2\xEF\xBC\xA3", ENGRAM_FOLD_COMPAT, "abc");   /* fullwidth ABC */
    norm_is("\xD0\x81\xD0\xBB\xD0\xBA\xD0\xB0", ENGRAM_FOLD_COMPAT, "\xD0\xB5\xD0\xBB\xD0\xBA\xD0\xB0");

    ET_SECTION("normalize: a DECOMPOSED accent folds too, and survives under CASE");
    norm_is("e\xCC\x81t\xC3\xA9", ENGRAM_FOLD_COMPAT, "ete");
    norm_is("e\xCC\x81", ENGRAM_FOLD_CASE, "e\xCC\x81");

    ET_SECTION("normalize: apostrophes vanish, so don't == dont");
    norm_is("Don't", ENGRAM_FOLD_COMPAT, "dont");
    norm_is("Don\xE2\x80\x99t", ENGRAM_FOLD_COMPAT, "dont");
    norm_is("a\xE2\x80\x8D" "b", ENGRAM_FOLD_COMPAT, "ab");                  /* ZWJ removed    */
    norm_is("a\xE2\x80\x8B" "b", ENGRAM_FOLD_COMPAT, "a b");                 /* ZWSP breaks    */

    ET_SECTION("normalize: Hindi keeps its vowel signs and virama under COMPAT");
    /* "hindi" in Devanagari: HA, VOWEL SIGN I, NA, VIRAMA, DA, VOWEL SIGN II. Stripping every
     * combining mark -- the common shortcut -- would reduce this to three consonants. */
    norm_is("\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80",
            ENGRAM_FOLD_COMPAT,
            "\xE0\xA4\xB9\xE0\xA4\xBF\xE0\xA4\xA8\xE0\xA5\x8D\xE0\xA4\xA6\xE0\xA5\x80");

    ET_SECTION("normalize: Chinese passes through unchanged, one IDEO per character");
    norm_is("\xE4\xB8\xAD\xE6\x96\x87", ENGRAM_FOLD_COMPAT, "\xE4\xB8\xAD\xE6\x96\x87");
    ET_OK(engram_textit_init(&it, "\xE4\xB8\xAD\xE6\x96\x87", 6u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT));
    for (n = 0; engram_textit_next(&it, &cp, &k); n++) ET_EQ_U64(k, ENGRAM_CLASS_IDEO);
    ET_EQ_U64(n, 2u);

    ET_SECTION("stream: STRICT refuses invalid UTF-8; REPLACE substitutes and COUNTS");
    ET_RC(engram_text_normalize("ok\xC0\xAF", 4u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT, &out, &len),
          ENGRAM_E_UTF8);
    ET_CHECK(out == NULL && len == 0u);
    ET_OK(engram_textit_init(&it, "ok\xC0\xAF!", 5u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE));
    for (n = 0; engram_textit_next(&it, &cp, &k); n++) {}
    ET_OK(it.err);
    ET_EQ_U64(it.n_invalid, 2u);
    ET_EQ_U64(n, 5u);                                  /* o k FFFD FFFD ! */

    ET_SECTION("stream: counters -- codepoints decoded, characters dropped");
    ET_OK(engram_textit_init(&it, "a'b\xCC\x81", 5u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT));
    for (n = 0; engram_textit_next(&it, &cp, &k); n++) {}
    ET_EQ_U64(it.n_codepoints, 4u);
    ET_EQ_U64(it.n_dropped, 2u);                       /* the apostrophe and the acute */
    ET_EQ_U64(n, 2u);

    ET_SECTION("stream: argument errors");
    ET_RC(engram_textit_init(NULL, "x", 1u, ENGRAM_FOLD_NONE, ENGRAM_UTF8_STRICT), ENGRAM_E_ARG);
    ET_RC(engram_textit_init(&it, NULL, 1u, ENGRAM_FOLD_NONE, ENGRAM_UTF8_STRICT), ENGRAM_E_ARG);
    ET_RC(engram_textit_init(&it, "x", 1u, (engram_fold)9, ENGRAM_UTF8_STRICT), ENGRAM_E_ARG);
    ET_RC(engram_textit_init(&it, "x", 1u, ENGRAM_FOLD_NONE, (engram_utf8_policy)9), ENGRAM_E_ARG);
    ET_RC(engram_text_normalize("x", 1u, ENGRAM_FOLD_NONE, ENGRAM_UTF8_STRICT, NULL, &len), ENGRAM_E_ARG);
}

static void test_normalize_faults(void)
{
    uint64_t k;
    unsigned refused = 0, ok = 0;
    const char *in = "Caf\xC3\xA9 au lait, s'il vous pla\xC3\xAEt -- ENGRAM";

    ET_SECTION("normalize: every allocation failure is a clean refusal, never a partial string");
    for (k = 1; k <= 64u; k++) {
        char *out = (char *)1;
        size_t len = 99u;
        engram_rc rc;
        engram_alloc_fail_at(k);
        rc = engram_text_normalize(in, strlen(in), ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT, &out, &len);
        engram_alloc_fail_at(0u);
        if (rc == ENGRAM_OK) {
            ok++;
            ET_CHECK(out && strcmp(out, "cafe au lait, sil vous plait -- engram") == 0);
            engram_free(out);
            break;                                      /* past the last allocation: done */
        }
        refused++;
        ET_CHECK(rc == ENGRAM_E_MEM && out == NULL && len == 0u);
    }
    ET_CHECK(refused >= 1u && ok == 1u);
    printf("       %u injected allocation failures refused cleanly before success\n", refused);
}

int main(void)
{
    printf("ENGRAM P1.2 -- text: UTF-8, classes, folding\n");
    ET_SELFTEST();
    test_utf8();
    test_ucd_exhaustive();
    test_standard_facts();
    test_normalize();
    test_normalize_faults();
    return et_report("test_text");
}
