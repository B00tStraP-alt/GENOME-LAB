/* ==================================================================================================
 * test_align.c -- P1.3 (part 2): the alignment. Every claim in engram_align.h, checked against a
 * definition that shares no code with the implementation.
 * ==================================================================================================
 *   DEFINITION   engram_align_dist equals the minimum, over EVERY stretch d[a..b) of the text, of the
 *                textbook optimal-string-alignment distance (full matrix, both borders counted),
 *                exhaustively on random strings over 2, 3 and 5 letters -- small alphabets, so matches,
 *                repeats and transpositions are everywhere
 *   BOUNDS       max(0, qn - dn) <= dist <= qn; 0 exactly when the cue occurs verbatim
 *   EDITS        each substitution, insertion, deletion and adjacent swap costs one edit, on hand-built
 *                cases, and on real text: fragments of every corpus align at 0, and k generated typos
 *                cost at most 2k (a swap is two Levenshtein edits, and OSA <= Levenshtein, which obeys
 *                the triangle inequality). How often it is <= k is reported.
 *   NORMALISED   case folded, space runs collapsed, ends trimmed, IGNOREs dropped; E_FULL at the cap;
 *                STRICT rejects ill-formed text; never more than FOLD_OUT_MAX codepoints per byte
 *   ALLOCATES    nothing
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_align.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"
#include "../src/engram_text.h"

#include <stdlib.h>
#include <string.h>

static const char *data_dir(void)
{
    const char *e = getenv("ENGRAM_TEST_DATA");
    return (e && e[0]) ? e : "../../test/data";
}

static int quick(void)
{
    const char *e = getenv("ENGRAM_TEST_QUICK");
    return e && e[0] == '1';
}

/* ---- the definition: textbook OSA between two whole strings, full matrix ---------------------- */
#define BF_MAX 16
static uint32_t osa_full(const uint32_t *a, size_t an, const uint32_t *b, size_t bn)
{
    uint32_t D[BF_MAX + 1][BF_MAX + 1];
    size_t i, j;
    for (i = 0; i <= an; i++) D[i][0] = (uint32_t)i;
    for (j = 0; j <= bn; j++) D[0][j] = (uint32_t)j;
    for (i = 1; i <= an; i++)
        for (j = 1; j <= bn; j++) {
            uint32_t v = D[i - 1][j - 1] + (a[i - 1] != b[j - 1]);
            if (D[i - 1][j] + 1u < v) v = D[i - 1][j] + 1u;
            if (D[i][j - 1] + 1u < v) v = D[i][j - 1] + 1u;
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] && D[i - 2][j - 2] + 1u < v)
                v = D[i - 2][j - 2] + 1u;
            D[i][j] = v;
        }
    return D[an][bn];
}

static uint32_t bf_dist(const uint32_t *q, size_t qn, const uint32_t *d, size_t dn)
{
    uint32_t best = (uint32_t)qn;
    size_t a, b;
    for (a = 0; a <= dn; a++)
        for (b = a; b <= dn; b++) {
            uint32_t v = osa_full(q, qn, d + a, b - a);
            if (v < best) best = v;
        }
    return best;
}

static int occurs(const uint32_t *q, size_t qn, const uint32_t *d, size_t dn)
{
    size_t a;
    if (qn == 0u) return 1;
    for (a = 0; a + qn <= dn; a++) if (!memcmp(d + a, q, qn * sizeof *q)) return 1;
    return 0;
}

static void test_definition(void)
{
    static const unsigned alpha[] = { 2u, 3u, 5u };
    uint32_t q[BF_MAX], d[BF_MAX], work[3 * (BF_MAX + 1)];
    engram_rng r;
    unsigned ai;
    unsigned long cases = quick() ? 20000ul : 200000ul, c, bad = 0, badb = 0, badz = 0;
    ET_SECTION("definition: equals min over every stretch of the textbook OSA distance (exhaustive stretches)");
    engram_rng_seed(&r, 0xA11C0DEull);
    for (ai = 0; ai < 3u; ai++)
        for (c = 0; c < cases / 3u; c++) {
            size_t qn = (size_t)engram_rng_below(&r, 9u), dn = (size_t)engram_rng_below(&r, 13u), i;
            uint32_t got, want;
            for (i = 0; i < qn; i++) q[i] = 'a' + (uint32_t)engram_rng_below(&r, alpha[ai]);
            for (i = 0; i < dn; i++) d[i] = 'a' + (uint32_t)engram_rng_below(&r, alpha[ai]);
            got = engram_align_dist(q, qn, d, dn, work);
            want = bf_dist(q, qn, d, dn);
            if (got != want && bad++ < 5u) printf("         qn %lu dn %lu got %u want %u\n",
                                                  (unsigned long)qn, (unsigned long)dn, got, want);
            if (got > qn || got + dn < qn) badb++;
            if ((got == 0u) != occurs(q, qn, d, dn)) badz++;
        }
    printf("       %lu random cases over 2, 3 and 5 letters\n", cases / 3u * 3u);
    ET_EQ_U64(bad, 0u);
    ET_SECTION("bounds: max(0, qn - dn) <= dist <= qn, and dist == 0 exactly when the cue occurs");
    ET_EQ_U64(badb, 0u);
    ET_EQ_U64(badz, 0u);
}

/* ---- hand-built edits ------------------------------------------------------------------------- */
static uint32_t dist_str(const char *q, const char *d)
{
    static uint32_t qc[256], dc[256], work[3 * 257];
    size_t qn = strlen(q), dn = strlen(d), i;
    for (i = 0; i < qn; i++) qc[i] = (uint8_t)q[i];
    for (i = 0; i < dn; i++) dc[i] = (uint8_t)d[i];
    return engram_align_dist(qc, qn, dc, dn, work);
}

static void test_edits(void)
{
    ET_SECTION("edits: verbatim, one substitution, insertion, deletion, adjacent swap; empty sides");
    ET_EQ_U64(dist_str("brown fox", "the quick brown fox jumps"), 0u);
    ET_EQ_U64(dist_str("brawn fox", "the quick brown fox jumps"), 1u);      /* substitute */
    ET_EQ_U64(dist_str("brownn fox", "the quick brown fox jumps"), 1u);     /* delete from the cue */
    ET_EQ_U64(dist_str("brwn fox", "the quick brown fox jumps"), 1u);       /* insert into the cue */
    ET_EQ_U64(dist_str("borwn fox", "the quick brown fox jumps"), 1u);      /* adjacent swap */
    ET_EQ_U64(dist_str("borwn fxo", "the quick brown fox jumps"), 2u);
    ET_EQ_U64(dist_str("", "anything"), 0u);
    ET_EQ_U64(dist_str("abc", ""), 3u);
    ET_EQ_U64(dist_str("", ""), 0u);
    ET_EQ_U64(dist_str("abcdef", "abc"), 3u);                               /* qn - dn */
    ET_EQ_U64(dist_str("xyz", "aaaaaaaaaa"), 3u);                           /* nothing in common: qn */
    ET_EQ_U64(dist_str("fox", "fox"), 0u);
    ET_EQ_U64(dist_str("jumps", "the quick brown fox jumps"), 0u);          /* at the very end */
    ET_EQ_U64(dist_str("the q", "the quick brown fox jumps"), 0u);          /* at the very start */
    ET_EQ_U64(dist_str("ab", "xbay"), 1u);                                  /* a swap inside the text */
    ET_EQ_U64(dist_str("ca", "abc"), 1u);                                   /* delete one: stretch "c" */
}

/* ---- normalisation ---------------------------------------------------------------------------- */
static int norm_is(const char *text, const char *want)
{
    uint32_t out[256];
    size_t len = 0, i, wn = strlen(want);
    if (engram_align_norm(text, strlen(text), ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 256u, &len) != ENGRAM_OK)
        return 0;
    if (len != wn) return 0;
    for (i = 0; i < wn; i++) if (out[i] != (uint8_t)want[i]) return 0;
    return 1;
}

static void test_norm(void)
{
    uint32_t out[64];
    size_t len = 99, i;
    engram_rng r;
    unsigned long over = 0;
    ET_SECTION("normalised: folded, space runs collapsed to one, ends trimmed, E_FULL at the cap");
    ET_CHECK(norm_is("  Hello,   WORLD \t\n ", "hello, world"));
    ET_CHECK(norm_is("", ""));
    ET_CHECK(norm_is(" \t\n ", ""));
    ET_CHECK(norm_is("a\xC2\xAD" "b", "ab"));                              /* U+00AD SOFT HYPHEN is IGNORE */
    ET_CHECK(norm_is("a\xE2\x80\x8B" "b", "a b"));                         /* U+200B is SPACE here (P1.2) */
    ET_CHECK(norm_is("Caf\xC3\xA9", "cafe"));                               /* COMPAT drops the accent */
    ET_RC(engram_align_norm("abcdef", 6u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 4u, &len), ENGRAM_E_FULL);
    ET_EQ_U64(len, 4u);
    ET_RC(engram_align_norm("abcd", 4u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 4u, &len), ENGRAM_OK);
    ET_EQ_U64(len, 4u);
    ET_RC(engram_align_norm("ab  cd", 6u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 4u, &len), ENGRAM_E_FULL);
    ET_RC(engram_align_norm("ab  c", 5u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 4u, &len), ENGRAM_OK);
    ET_EQ_U64(len, 4u);
    ET_SECTION("normalised: STRICT refuses ill-formed text, REPLACE does not; bad arguments");
    ET_RC(engram_align_norm("a\xFF" "b", 3u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_STRICT, out, 64u, &len), ENGRAM_E_UTF8);
    ET_RC(engram_align_norm("a\xFF" "b", 3u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 64u, &len), ENGRAM_OK);
    ET_RC(engram_align_norm("a", 1u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 64u, NULL), ENGRAM_E_ARG);
    ET_RC(engram_align_norm("a", 1u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, NULL, 64u, &len), ENGRAM_E_ARG);
    ET_RC(engram_align_norm(NULL, 1u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 64u, &len), ENGRAM_E_ARG);
    ET_RC(engram_align_norm(NULL, 0u, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, out, 64u, &len), ENGRAM_OK);
    ET_EQ_U64(len, 0u);
    ET_SECTION("normalised: random bytes never yield more than FOLD_OUT_MAX codepoints per byte");
    engram_rng_seed(&r, 0xB0B0ull);
    for (i = 0; i < (quick() ? 2000u : 20000u); i++) {
        static uint8_t b[16];
        static uint32_t big[16 * ENGRAM_FOLD_OUT_MAX];
        size_t n = (size_t)engram_rng_below(&r, 17u), k;
        for (k = 0; k < n; k++) b[k] = (uint8_t)engram_rng_below(&r, 256u);
        if (engram_align_norm(b, n, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE, big, n * ENGRAM_FOLD_OUT_MAX, &len)
            != ENGRAM_OK) over++;
    }
    ET_EQ_U64(over, 0u);
}

/* ---- real text: fragments align at 0; k typos cost at most 2k --------------------------------- */
static void test_corpora(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    static uint32_t d[8192], q[8192], work[3 * 8193];
    unsigned li;
    uint64_t calls;
    unsigned long frag = 0, frag_bad = 0, typo = 0, typo_bad = 0, typo_le_k = 0;
    engram_rng r;
    ET_SECTION("real text: fragments of every corpus align at 0; k typos cost <= 2k; nothing allocated");
    engram_rng_seed(&r, 0x7E47ull);
    calls = engram_alloc_calls();
    for (li = 0; li < 5u; li++) {
        char name[64], *path;
        uint8_t *buf = NULL;
        size_t n = 0, i;
        snprintf(name, sizeof name, "corpus_%s.txt", langs[li]);
        path = engram_path_join(data_dir(), name);
        if (!path || engram_file_read(path, &buf, &n) != ENGRAM_OK) {
            ET_CHECKF(0, "%s did not load", name); engram_free(path); continue;
        }
        engram_free(path);
        calls = engram_alloc_calls();
        for (i = 0; i < n; ) {                      /* each line: normalise, cut fragments, perturb */
            size_t e = i, dn = 0, t;
            while (e < n && buf[e] != '\n') e++;
            if (e - i < 2048u && engram_align_norm(buf + i, e - i, ENGRAM_FOLD_COMPAT, ENGRAM_UTF8_REPLACE,
                                                   d, 8192u, &dn) == ENGRAM_OK && dn >= 8u)
                for (t = 0; t < 4u; t++) {
                    size_t len = 4u + (size_t)engram_rng_below(&r, (uint64_t)(dn / 2u - 1u));
                    size_t at = (size_t)engram_rng_below(&r, (uint64_t)(dn - len + 1u)), qn = len, k, x;
                    unsigned kk = (unsigned)engram_rng_below(&r, 4u);
                    memcpy(q, d + at, len * sizeof *q);
                    frag++;
                    if (engram_align_dist(q, qn, d, dn, work) != 0u) frag_bad++;
                    for (k = 0; k < kk; k++) {      /* substitute, delete, insert or swap */
                        size_t p = (size_t)engram_rng_below(&r, (uint64_t)qn);
                        unsigned op = (unsigned)engram_rng_below(&r, 4u);
                        if (op == 0u) q[p] = 0x2603u;
                        else if (op == 1u && qn > 1u) { memmove(q + p, q + p + 1u, (qn - p - 1u) * sizeof *q); qn--; }
                        else if (op == 2u) { memmove(q + p + 1u, q + p, (qn - p) * sizeof *q); q[p] = 0x2603u; qn++; }
                        else if (p + 1u < qn) { uint32_t tmp = q[p]; q[p] = q[p + 1u]; q[p + 1u] = tmp; }
                    }
                    x = engram_align_dist(q, qn, d, dn, work);
                    typo++;
                    if (x > 2u * kk) typo_bad++;
                    if (x <= kk) typo_le_k++;
                }
            i = e + 1u;
        }
        ET_EQ_U64(engram_alloc_calls(), calls);
        engram_free(buf);
    }
    printf("       %lu fragments, %lu perturbed with 0-3 edits; %.2f%% of those cost <= k edits\n",
           frag, typo, typo ? 100.0 * (double)typo_le_k / (double)typo : 0.0);
    ET_CHECK(frag > 1000u);
    ET_EQ_U64(frag_bad, 0u);
    ET_EQ_U64(typo_bad, 0u);
    ET_CHECK(typo_le_k * 100u >= typo * 95u);
}

int main(void)
{
    printf("ENGRAM P1.3 -- alignment\n");
    ET_SELFTEST();
    test_definition();
    test_edits();
    test_norm();
    test_corpora();
    return et_report("test_align");
}
