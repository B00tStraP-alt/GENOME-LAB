/* ==================================================================================================
 * test_chunk.c -- P1.3 (part 1): the episode cutter. Every guarantee in engram_chunk.h, checked on
 * every corpus and on inputs built to break it.
 * ==================================================================================================
 *   BOUNDED     no chunk exceeds cap
 *   WHOLE       every chunk begins and ends on a decoder step of the whole text
 *   COVERING    every byte of every non-whitespace codepoint lies in some chunk
 *   PROGRESS    chunk starts strictly increase; the iterator terminates
 *   NO FLUFF    no chunk begins or ends with whitespace
 *   DISJOINT    with overlap 0, chunks do not overlap and come in order
 *   OVERLAP     with overlap > 0, each chunk after the first begins at a word start inside the
 *               previous chunk, or where the previous one ended
 *   ALLOCATES   nothing
 * plus behaviour on hand-built cases (sentence preference, "3.14", CJK full stops), and the cut-kind
 * distribution on real text, reported.
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_chunk.h"
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

typedef struct { size_t off, len; } span;

/* Chunk the whole text; verify every guarantee; return the number of chunks or -1 on failure. */
static long check_all(const uint8_t *p, size_t n, size_t cap, size_t overlap, const char *what,
                      engram_chunkit *out_it)
{
    uint8_t *step = (uint8_t *)engram_array(n + 1u, 1u);    /* 1 at every decoder-step boundary   */
    uint8_t *content = (uint8_t *)engram_array(n + 1u, 1u); /* 1 on bytes of non-space codepoints */
    uint8_t *covered = (uint8_t *)engram_array(n + 1u, 1u);
    engram_chunkit it;
    size_t off, len, pos, prev_off = 0, prev_end = 0, k;
    long count = 0;
    int bad_bound = 0, bad_whole = 0, bad_prog = 0, bad_fluff = 0, bad_disj = 0, bad_ovl = 0, bad_cov = 0;
    uint64_t calls;
    if (!step || !content || !covered) { engram_free(step); engram_free(content); engram_free(covered); return -1; }
    memset(step, 0, n + 1u); memset(content, 0, n + 1u); memset(covered, 0, n + 1u);
    for (pos = 0; pos < n; ) {
        uint32_t cp; int v;
        size_t u = engram_utf8_decode(p + pos, n - pos, &cp, &v);
        step[pos] = 1;
        if (engram_cp_class(cp) != ENGRAM_CLASS_SPACE) memset(content + pos, 1, u);
        pos += u;
    }
    step[n] = 1;
    if (engram_chunkit_init(&it, p, n, cap, overlap) != ENGRAM_OK) { count = -1; goto out; }
    calls = engram_alloc_calls();
    while (engram_chunkit_next(&it, &off, &len)) {
        size_t end = off + len;
        uint32_t cp; int v;
        count++;
        if (len == 0 || len > cap || end > n) { bad_bound = 1; break; }
        if (!step[off] || !step[end]) bad_whole = 1;
        if (count > 1 && off <= prev_off) { bad_prog = 1; break; }
        (void)engram_utf8_decode(p + off, n - off, &cp, &v);
        if (engram_cp_class(cp) == ENGRAM_CLASS_SPACE) bad_fluff = 1;
        {   /* the last codepoint of the chunk: step back to the last boundary before end */
            size_t q = end - 1u;
            while (q > off && !step[q]) q--;
            (void)engram_utf8_decode(p + q, n - q, &cp, &v);
            if (engram_cp_class(cp) == ENGRAM_CLASS_SPACE) bad_fluff = 1;
        }
        if (overlap == 0u && count > 1 && off < prev_end) bad_disj = 1;
        if (overlap > 0u && count > 1) {
            if (off > prev_end) {                        /* a gap is only whitespace, checked below */
            } else if (off < prev_end) {
                uint32_t pc; int pv; size_t q = off - 1u;
                while (q > prev_off && !step[q]) q--;
                (void)engram_utf8_decode(p + q, n - q, &pc, &pv);
                if (engram_cp_class(pc) != ENGRAM_CLASS_SPACE) bad_ovl = 1;   /* not a word start */
            }
        }
        memset(covered + off, 1, len);
        prev_off = off;
        prev_end = end;
        if ((size_t)count > n + 1u) { bad_prog = 1; break; }             /* cannot terminate */
    }
    if (engram_alloc_calls() != calls) bad_bound = 1;
    for (k = 0; k < n; k++) if (content[k] && !covered[k]) { bad_cov = 1; break; }
    ET_CHECKF(!bad_bound, "%s cap=%lu ovl=%lu: a chunk broke its bound (or the cutter allocated)", what,
              (unsigned long)cap, (unsigned long)overlap);
    ET_CHECKF(!bad_whole, "%s cap=%lu: a chunk split a decoder step", what, (unsigned long)cap);
    ET_CHECKF(!bad_prog, "%s cap=%lu: starts did not strictly increase", what, (unsigned long)cap);
    ET_CHECKF(!bad_fluff, "%s cap=%lu: a chunk begins or ends with whitespace", what, (unsigned long)cap);
    ET_CHECKF(!bad_disj, "%s cap=%lu: overlap 0 but chunks overlap", what, (unsigned long)cap);
    ET_CHECKF(!bad_ovl, "%s cap=%lu ovl=%lu: an overlapping chunk does not begin at a word start", what,
              (unsigned long)cap, (unsigned long)overlap);
    ET_CHECKF(!bad_cov, "%s cap=%lu ovl=%lu: content byte %lu is in no chunk", what, (unsigned long)cap,
              (unsigned long)overlap, (unsigned long)k);
    if (out_it) *out_it = it;
out:
    engram_free(step); engram_free(content); engram_free(covered);
    return count;
}

static void expect_chunks(const char *text, size_t cap, const char *const *want, size_t nwant)
{
    engram_chunkit it;
    size_t off, len, k = 0;
    ET_OK(engram_chunkit_init(&it, text, strlen(text), cap, 0u));
    while (engram_chunkit_next(&it, &off, &len)) {
        if (k < nwant)
            ET_CHECKF(len == strlen(want[k]) && memcmp(text + off, want[k], len) == 0,
                      "chunk %lu is \"%.*s\", expected \"%s\"", (unsigned long)k, (int)len, text + off, want[k]);
        k++;
    }
    ET_EQ_U64(k, nwant);
}

static void test_contract(void)
{
    engram_chunkit it;
    size_t off = 7, len = 7;
    ET_SECTION("contract: refusals");
    ET_RC(engram_chunkit_init(NULL, "a", 1u, 64u, 0u), ENGRAM_E_ARG);
    ET_RC(engram_chunkit_init(&it, NULL, 1u, 64u, 0u), ENGRAM_E_ARG);
    ET_RC(engram_chunkit_init(&it, "a", 1u, ENGRAM_CHUNK_MIN_CAP - 1u, 0u), ENGRAM_E_ARG);
    ET_RC(engram_chunkit_init(&it, "a", 1u, ENGRAM_EPI_TAIL + 1u, 0u), ENGRAM_E_ARG);
    ET_RC(engram_chunkit_init(&it, "a", 1u, 64u, 32u), ENGRAM_E_ARG);         /* overlap >= cap/2 */
    ET_OK(engram_chunkit_init(&it, "a", 1u, 64u, 31u));
    ET_CHECK(engram_chunkit_next(&it, NULL, &len) == 0);
    ET_CHECK(engram_chunkit_next(NULL, &off, &len) == 0);

    ET_SECTION("contract: empty and blank text yield nothing");
    ET_OK(engram_chunkit_init(&it, "", 0u, 64u, 0u));
    ET_CHECK(engram_chunkit_next(&it, &off, &len) == 0);
    ET_OK(engram_chunkit_init(&it, NULL, 0u, 64u, 0u));
    ET_CHECK(engram_chunkit_next(&it, &off, &len) == 0);
    ET_OK(engram_chunkit_init(&it, " \n\t\n  ", 6u, 64u, 0u));
    ET_CHECK(engram_chunkit_next(&it, &off, &len) == 0);

    ET_SECTION("behaviour: a sentence end is preferred to a word break");
    {
        static const char *const w[] = { "The first sentence is here.", "The second one follows it",
                                         "closely." };
        expect_chunks("The first sentence is here. The second one follows it closely.", 32u, w, 3u);
    }
    ET_SECTION("behaviour: the full stop in 3.14 is not a sentence end");
    {
        static const char *const w[] = { "Pi is 3.14 and e is 2.71 which", "are numbers." };
        expect_chunks("Pi is 3.14 and e is 2.71 which are numbers.", 32u, w, 2u);
    }
    ET_SECTION("behaviour: a closing quote stays with its sentence");
    {
        static const char *const w[] = { "He said \"stop it now.\"", "Then he left the room quietly." };
        expect_chunks("He said \"stop it now.\" Then he left the room quietly.", 32u, w, 2u);
    }
    ET_SECTION("behaviour: CJK full stops cut without spaces");
    {
        /* 大家又說了一回閒話． (10 x 3 bytes) then 至晚飯后又往賈母處來請安． */
        static const char *const w[] = { "\xE5\xA4\xA7\xE5\xAE\xB6\xE5\x8F\x88\xE8\xAA\xAA\xE4\xBA\x86\xE4\xB8\x80"
                                         "\xE5\x9B\x9E\xE9\x96\x92\xE8\xA9\xB1\xEF\xBC\x8E",
                                         "\xE8\x87\xB3\xE6\x99\x9A\xE9\xA3\xAF\xE5\x90\x8E\xE5\x8F\x88\xE5\xBE\x80"
                                         "\xE8\xB3\x88\xE6\xAF\x8D\xE8\x99\x95\xE4\xBE\x86\xE8\xAB\x8B\xE5\xAE\x89"
                                         "\xEF\xBC\x8E" };
        char t[256];
        snprintf(t, sizeof t, "%s%s", w[0], w[1]);
        expect_chunks(t, 40u, w, 2u);
    }
    ET_SECTION("behaviour: a paragraph break is preferred to a later sentence end");
    {
        static const char *const w[] = { "Short paragraph one", "Next paragraph. It goes on." };
        expect_chunks("Short paragraph one\n\nNext paragraph. It goes on.", 40u, w, 2u);
    }
}

static void test_hostile(void)
{
    static const size_t caps[] = { 32u, 64u, 240u, 480u };
    uint8_t *b;
    size_t n = 200000u, i, ci;
    engram_rng r;

    ET_SECTION("hostile: random bytes, including ill-formed UTF-8, every cap, with and without overlap");
    b = (uint8_t *)engram_malloc(n);
    ET_CHECK(b != NULL);
    if (!b) return;
    engram_rng_seed(&r, 0xC4u);
    for (i = 0; i < n; i++) b[i] = (uint8_t)engram_rng_below(&r, 256u);
    for (ci = 0; ci < 4u; ci++) {
        ET_CHECK(check_all(b, n, caps[ci], 0u, "random", NULL) > 0);
        ET_CHECK(check_all(b, n, caps[ci], caps[ci] / 4u, "random", NULL) > 0);
    }

    ET_SECTION("hostile: no spaces at all, only spaces, only newlines, one long whitespace run");
    memset(b, 'x', n);
    for (ci = 0; ci < 4u; ci++) ET_CHECK(check_all(b, n, caps[ci], 0u, "no-spaces", NULL) > 0);
    memset(b, ' ', n);
    for (ci = 0; ci < 4u; ci++) ET_CHECK(check_all(b, n, caps[ci], 0u, "all-spaces", NULL) == 0);
    memset(b, '\n', n);
    ET_CHECK(check_all(b, n, 64u, 0u, "all-newlines", NULL) == 0);
    for (i = 0; i < n; i++) b[i] = (uint8_t)((i % 3u == 0u) ? '\n' : ' ');
    memcpy(b, "one two three four five six seven eight\n", 40);
    memcpy(b + n - 12u, "\nlast words.", 12);
    ET_CHECK(check_all(b, n, 64u, 0u, "whitespace-run", NULL) >= 2);

    ET_SECTION("hostile: exact sizes -- cap bytes, cap + 1, one codepoint, a 4-byte codepoint at the edge");
    for (ci = 0; ci < 4u; ci++) {
        memset(b, 'a', caps[ci] + 1u);
        ET_CHECK(check_all(b, caps[ci], caps[ci], 0u, "exact-cap", NULL) == 1);
        ET_CHECK(check_all(b, caps[ci] + 1u, caps[ci], 0u, "cap-plus-one", NULL) == 2);
    }
    ET_CHECK(check_all((const uint8_t *)"z", 1u, 32u, 0u, "one", NULL) == 1);
    for (i = 0; i + 4u <= 64u; i += 4u) memcpy(b + i, "\xF0\x9F\x98\x80", 4u);   /* U+1F600 x 16 */
    ET_CHECK(check_all(b, 64u, 33u, 0u, "4-byte-edge", NULL) == 2);
    engram_free(b);
}

static void test_corpora(void)
{
    static const char *langs[] = { "en", "fr", "de", "sv", "zh" };
    static const size_t caps[] = { 240u, 480u };
    unsigned li, ci;
    ET_SECTION("corpora: every language, paragraphs rejoined with blank lines, cap 240 and 480");
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
        {   /* one line per paragraph in the file: make them real paragraphs */
            uint8_t *t = (uint8_t *)engram_malloc(2u * n + 1u);
            size_t m = 0;
            if (!t) { ET_CHECK(0); engram_free(buf); continue; }
            for (i = 0; i < n; i++) { t[m++] = buf[i]; if (buf[i] == '\n') t[m++] = '\n'; }
            for (ci = 0; ci < 2u; ci++) {
                engram_chunkit it;
                long c0 = check_all(t, m, caps[ci], 0u, langs[li], &it);
                long c1 = check_all(t, m, caps[ci], caps[ci] / 8u, langs[li], NULL);
                ET_CHECK(c0 > 0 && c1 >= c0);
                printf("       %s cap %3lu: %5ld chunks, mean %5.1f B | cut at paragraph %4.1f%%, sentence"
                       " %4.1f%%, space %4.1f%%, hard %4.1f%%\n", langs[li], (unsigned long)caps[ci], c0,
                       c0 > 0 ? (double)m / (double)c0 : 0.0,
                       100.0 * (double)it.n_paragraph / (double)(c0 > 1 ? c0 - 1 : 1),
                       100.0 * (double)it.n_sentence / (double)(c0 > 1 ? c0 - 1 : 1),
                       100.0 * (double)it.n_space / (double)(c0 > 1 ? c0 - 1 : 1),
                       100.0 * (double)it.n_hard / (double)(c0 > 1 ? c0 - 1 : 1));
            }
            engram_free(t);
        }
        engram_free(buf);
    }
}

int main(void)
{
    printf("ENGRAM P1.3 -- episode cutter\n");
    ET_SELFTEST();
    test_contract();
    test_hostile();
    test_corpora();
    return et_report("test_chunk");
}
