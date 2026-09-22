/* ==================================================================================================
 * test_core.c -- P1.1: the core runtime, proven.
 * ==================================================================================================
 * Every pinned value below was computed by an INDEPENDENT implementation (tools/ref_core.py), not by
 * running this code and copying its output -- a test that pins a function's output against itself
 * proves only that the function is deterministic, not that it is right. mix64(0) is additionally
 * anchored to the published splitmix64 first output for seed 0.
 * ============================================================================================== */
#include "engram_test.h"

#include "../src/engram_buf.h"
#include "../src/engram_plat.h"
#include "../src/engram_rng.h"

#include <stdlib.h>
#include <string.h>

#define SCRATCH_DIR "engram_test_scratch"

/* ==================================================================================================
 * RETURN CODES
 * ============================================================================================== */
static void test_rc(void)
{
    int v, w;
    ET_SECTION("return codes: every code named, every name distinct");
    for (v = (int)ENGRAM_E_INTERNAL; v <= (int)ENGRAM_OK; v++) {
        const char *n = engram_rcname((engram_rc)v);
        const char *m = engram_strerror((engram_rc)v);
        ET_CHECKF(n && strcmp(n, "ENGRAM_E_UNKNOWN") != 0, "code %d has no name", v);
        ET_CHECKF(m && strcmp(m, "unknown error code") != 0, "code %d has no message", v);
        for (w = (int)ENGRAM_E_INTERNAL; w < v; w++)
            ET_CHECKF(strcmp(n, engram_rcname((engram_rc)w)) != 0, "codes %d and %d share a name", v, w);
    }
    ET_STREQ(engram_rcname(ENGRAM_OK), "ENGRAM_OK");
    ET_STREQ(engram_rcname(ENGRAM_E_AUTH), "ENGRAM_E_AUTH");

    ET_SECTION("return codes: an unknown code is reported, never NULL");
    ET_STREQ(engram_strerror((engram_rc)-999), "unknown error code");
    ET_STREQ(engram_rcname((engram_rc)12345), "ENGRAM_E_UNKNOWN");
}

/* ==================================================================================================
 * HASHING
 * ============================================================================================== */
static void test_hash(void)
{
    uint64_t a, b;
    unsigned i, bits;
    long total = 0;

    ET_SECTION("mix64: pinned to an independent implementation");
    ET_EQ_U64(engram_mix64(0u),                    0xE220A8397B1DCDAFull);   /* = published splitmix64 */
    ET_EQ_U64(engram_mix64(1u),                    0x910A2DEC89025CC1ull);
    ET_EQ_U64(engram_mix64(0xFFFFFFFFFFFFFFFFull), 0xE4D971771B652C20ull);
    ET_EQ_U64(engram_mix64(0x0123456789ABCDEFull), 0x157A3807A48FAA9Dull);

    ET_SECTION("mix2: pinned, and order-sensitive");
    ET_EQ_U64(engram_mix2(1u, 2u), 0x01CE0A687AADA258ull);
    ET_EQ_U64(engram_mix2(2u, 1u), 0xCB99105A710440B5ull);
    ET_CHECK(engram_mix2(1u, 2u) != engram_mix2(2u, 1u));

    ET_SECTION("hash_bytes: pinned, seed- and length-sensitive");
    ET_EQ_U64(engram_hash_bytes("", 0u, 0u),     0x5B21F68FFA77F14Cull);
    ET_EQ_U64(engram_hash_bytes("abc", 3u, 0u),  0xE6D7AF2DB17CA299ull);
    ET_EQ_U64(engram_hash_bytes("abc", 3u, 1u),  0xFD5F76974069E6A4ull);
    ET_EQ_U64(engram_hash_bytes("a\0", 2u, 0u),  0xE634F2798162602Aull);
    ET_CHECK(engram_hash_bytes("a", 1u, 0u) != engram_hash_bytes("a\0", 2u, 0u));

    ET_SECTION("mix64: avalanche -- one flipped input bit flips about half the output");
    for (i = 0; i < 64u; i++) {
        a = engram_mix64(0x243F6A8885A308D3ull);
        b = engram_mix64(0x243F6A8885A308D3ull ^ ((uint64_t)1u << i));
        a ^= b;                                       /* the bits that differ */
        for (bits = 0; a; a &= a - 1u) bits++;        /* Kernighan popcount   */
        ET_CHECKF(bits >= 16u && bits <= 48u, "bit %u flipped only %u output bits", i, bits);
        total += (long)bits;
    }
    /* Mean over 64 single-bit flips must sit near 32; a poor mixer drifts far from it. */
    ET_CHECKF(total >= 64L * 28L && total <= 64L * 36L, "mean flipped bits %ld/64", total / 64L);
}

/* ==================================================================================================
 * BUILD INFO
 * ============================================================================================== */
static void test_build_info(void)
{
    engram_build_info bi;
    ET_SECTION("build info reports the geometry this binary was compiled at");
    ET_RC(engram_build_info_get(NULL), ENGRAM_E_ARG);
    ET_OK(engram_build_info_get(&bi));
    ET_EQ_U64(bi.major, ENGRAM_VER_MAJOR);
    ET_EQ_U64(bi.minor, ENGRAM_VER_MINOR);
    ET_EQ_U64(bi.d, ENGRAM_D);
    ET_EQ_U64(bi.epi_e, ENGRAM_EPI_E);
    ET_EQ_U64(bi.epi_k, ENGRAM_EPI_K);
    ET_STREQ(bi.string, ENGRAM_VER_STRING);
    ET_CHECK(bi.platform && bi.platform[0]);
    ET_CHECK(bi.compiler && bi.compiler[0]);
    printf("       %s on %s, built by %s\n", bi.string, bi.platform, bi.compiler);
}

/* ==================================================================================================
 * ALLOCATOR
 * ============================================================================================== */
static void test_alloc_basic(void)
{
    engram_alloc_stats s0, s1;
    unsigned char *p, *q;
    size_t i;
    char *s;

    ET_SECTION("alloc: malloc(0) is a real pointer, so NULL always means failure");
    engram_alloc_stats_get(&s0);
    p = (unsigned char *)engram_malloc(0);
    ET_CHECK(p != NULL);
    ET_EQ_U64(engram_alloc_size(p), 0u);
    ET_OK(engram_alloc_check(p));
    engram_free(p);

    ET_SECTION("alloc: live accounting returns exactly to where it started");
    p = (unsigned char *)engram_malloc(1000u);
    ET_CHECK(p != NULL);
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.live_blocks, s0.live_blocks + 1u);
    ET_EQ_U64(s1.live_bytes, s0.live_bytes + 1000u);
    ET_EQ_U64(engram_alloc_size(p), 1000u);
    engram_free(p);
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.live_blocks, s0.live_blocks);
    ET_EQ_U64(s1.live_bytes, s0.live_bytes);

    ET_SECTION("alloc: returned pointers are 16-byte aligned");
    for (i = 0; i < 64u; i++) {
        p = (unsigned char *)engram_malloc(i * 7u + 1u);
        ET_CHECKF(((uintptr_t)p & 15u) == 0u, "size %u misaligned", (unsigned)(i * 7u + 1u));
        engram_free(p);
    }

    ET_SECTION("alloc: calloc zeroes, and refuses an overflowing product");
    p = (unsigned char *)engram_calloc(33u, 31u);
    ET_CHECK(p != NULL);
    for (i = 0; p && i < 33u * 31u; i++) if (p[i]) break;
    ET_CHECK(i == 33u * 31u);
    engram_free(p);
    ET_CHECK(engram_calloc(SIZE_MAX / 2u + 1u, 3u) == NULL);
    ET_CHECK(engram_array(SIZE_MAX / 4u + 1u, 8u) == NULL);
    ET_CHECK(engram_malloc(SIZE_MAX - 4u) == NULL);

    ET_SECTION("alloc: realloc preserves contents across grow and shrink");
    p = (unsigned char *)engram_malloc(16u);
    for (i = 0; i < 16u; i++) p[i] = (unsigned char)(i * 11u);
    q = (unsigned char *)engram_realloc(p, 100000u);
    ET_CHECK(q != NULL);
    for (i = 0; q && i < 16u; i++) ET_CHECK(q[i] == (unsigned char)(i * 11u));
    ET_EQ_U64(engram_alloc_size(q), 100000u);
    p = (unsigned char *)engram_realloc(q, 8u);
    ET_CHECK(p != NULL);
    for (i = 0; p && i < 8u; i++) ET_CHECK(p[i] == (unsigned char)(i * 11u));
    ET_OK(engram_alloc_check(p));

    ET_SECTION("alloc: realloc(p, 0) is refused and p stays valid");
    ET_CHECK(engram_realloc(p, 0u) == NULL);
    ET_OK(engram_alloc_check(p));
    ET_CHECK(p[3] == (unsigned char)33u);
    engram_free(p);

    ET_SECTION("alloc: realloc(NULL, n) is malloc");
    p = (unsigned char *)engram_realloc(NULL, 24u);
    ET_CHECK(p != NULL);
    ET_EQ_U64(engram_alloc_size(p), 24u);
    engram_free(p);

    ET_SECTION("alloc: strdup / strndup");
    s = engram_strdup("engram");
    ET_STREQ(s, "engram");
    engram_free(s);
    s = engram_strndup("consolidate", 6u);
    ET_STREQ(s, "consol");
    engram_free(s);
    s = engram_strndup("ab", 99u);
    ET_STREQ(s, "ab");
    engram_free(s);
    ET_CHECK(engram_strdup(NULL) == NULL);
    engram_free(NULL);                                /* a no-op, and must not crash */
}

static void test_alloc_grow(void)
{
    uint32_t *a = NULL;
    size_t cap = 0, i;
    void *saved;
    size_t saved_cap;

    ET_SECTION("grow: doubles, preserves contents, never loses the array on failure");
    ET_OK(engram_grow((void **)&a, &cap, 5u, sizeof *a));
    ET_EQ_U64(cap, 8u);
    for (i = 0; i < 8u; i++) a[i] = (uint32_t)(i + 1u);
    ET_OK(engram_grow((void **)&a, &cap, 9u, sizeof *a));
    ET_EQ_U64(cap, 16u);
    for (i = 0; i < 8u; i++) ET_CHECK(a[i] == (uint32_t)(i + 1u));
    ET_OK(engram_grow((void **)&a, &cap, 3u, sizeof *a));      /* already big enough: unchanged */
    ET_EQ_U64(cap, 16u);

    saved = a;
    saved_cap = cap;
    engram_alloc_fail_at(1u);
    ET_RC(engram_grow((void **)&a, &cap, 1000u, sizeof *a), ENGRAM_E_MEM);
    ET_CHECK((void *)a == saved);
    ET_EQ_U64(cap, saved_cap);
    for (i = 0; i < 8u; i++) ET_CHECK(a[i] == (uint32_t)(i + 1u));
    ET_RC(engram_grow((void **)&a, &cap, SIZE_MAX / 2u, sizeof *a), ENGRAM_E_OVERFLOW);
    ET_RC(engram_grow(NULL, &cap, 4u, 4u), ENGRAM_E_ARG);
    ET_RC(engram_grow((void **)&a, &cap, 4u, 0u), ENGRAM_E_ARG);
    engram_free(a);
}

static void test_alloc_faults(void)
{
    engram_alloc_stats s0, s1;
    void *p, *q, *r;

    ET_SECTION("fault injection: the Nth allocation fails, exactly once, then all is normal");
    engram_alloc_stats_get(&s0);
    engram_alloc_fail_at(3u);
    p = engram_malloc(10u);
    q = engram_malloc(10u);
    r = engram_malloc(10u);                          /* the third: must fail */
    ET_CHECK(p != NULL);
    ET_CHECK(q != NULL);
    ET_CHECK(r == NULL);
    engram_free(p); engram_free(q);
    p = engram_malloc(10u);                          /* and the world behaves again */
    ET_CHECK(p != NULL);
    engram_free(p);
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.n_fail_injected, s0.n_fail_injected + 1u);

    ET_SECTION("fault injection: an overflow refusal still ticks exactly once (I3)");
    engram_alloc_fail_at(2u);
    ET_CHECK(engram_calloc(SIZE_MAX, 2u) == NULL);   /* tick 1: refused for overflow */
    ET_CHECK(engram_malloc(4u) == NULL);             /* tick 2: the injected one     */
    p = engram_malloc(4u);
    ET_CHECK(p != NULL);
    engram_free(p);

    ET_SECTION("fault injection: fail_at(0) disables");
    engram_alloc_fail_at(1u);
    engram_alloc_fail_at(0u);
    p = engram_malloc(4u);
    ET_CHECK(p != NULL);
    engram_free(p);
}

static void test_alloc_detectors(void)
{
    engram_alloc_stats s0, s1;
    unsigned char *p;
    unsigned char arena[64];
    size_t i;
    int all_dd;

    ET_SECTION("detector: a one-byte overrun is caught at free, and the block is not released");
    engram_alloc_stats_get(&s0);
    p = (unsigned char *)engram_malloc(40u);
    p[40] = 0x5Au;                                   /* one byte past the end: the planted defect */
    ET_RC(engram_alloc_check(p), ENGRAM_E_INTERNAL);
    engram_free(p);
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.n_overrun, s0.n_overrun + 1u);
    ET_PLANT_CORRUPTION(1u);
    ET_PLANT_LEAK(1u, 40u);                          /* leaked on purpose (I1) */

    ET_SECTION("detector: a double free inside the quarantine window is identified exactly");
    engram_alloc_stats_get(&s0);
    p = (unsigned char *)engram_malloc(24u);
    engram_free(p);
    engram_free(p);                                  /* the planted defect */
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.n_double_free, s0.n_double_free + 1u);
    ET_EQ_U64(s1.n_free, s0.n_free + 1u);            /* only ONE real free happened */
    ET_PLANT_CORRUPTION(1u);

    ET_SECTION("detector: a pointer that never came from here is refused");
    engram_alloc_stats_get(&s0);
    memset(arena, 0, sizeof arena);
    engram_free(arena + 32);                         /* header bytes 16..31 are zero: not ours */
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.n_foreign_free, s0.n_foreign_free + 1u);
    ET_PLANT_CORRUPTION(1u);

    ET_SECTION("quarantine: a freed block is poisoned, so stale reads see garbage");
    p = (unsigned char *)engram_malloc(64u);
    memset(p, 0x11, 64u);
    engram_free(p);
    /* Well-defined: the block is still owned by this allocator (in quarantine), never released. */
    for (all_dd = 1, i = 0; i < 64u; i++) if (p[i] != 0xDDu) all_dd = 0;
    ET_CHECK(all_dd);

    ET_SECTION("quarantine: holds blocks, and flush releases every one");
    engram_alloc_stats_get(&s0);
    ET_CHECK(s0.quarantine_blocks > 0u);
    engram_alloc_quarantine_flush();
    engram_alloc_stats_get(&s1);
    ET_EQ_U64(s1.quarantine_blocks, 0u);
    ET_EQ_U64(s1.quarantine_bytes, 0u);
}

/* ==================================================================================================
 * PLATFORM
 * ============================================================================================== */
static void test_clock(void)
{
    uint64_t t0, t1, prev;
    int64_t w;
    char buf[32];
    int i, mono = 1;

    ET_SECTION("clock: monotonic never goes backwards");
    prev = engram_now_ns();
    for (i = 0; i < 20000; i++) {
        t1 = engram_now_ns();
        if (t1 < prev) mono = 0;
        prev = t1;
    }
    ET_CHECK(mono);
    t0 = engram_now_ns();
    ET_CHECK(t0 > 0u);

    ET_SECTION("clock: wall time is plausible (after 2020, before 2200)");
    w = engram_wall_ms();
    ET_CHECK(w > 1577836800000ll);
    ET_CHECK(w < 7258118400000ll);

    ET_SECTION("wall_format: epoch, leap day, pre-epoch, far future, bounds");
    ET_OK(engram_wall_format(0, buf, sizeof buf));
    ET_STREQ(buf, "1970-01-01T00:00:00.000Z");
    ET_OK(engram_wall_format(951782400000ll, buf, sizeof buf));
    ET_STREQ(buf, "2000-02-29T00:00:00.000Z");
    ET_OK(engram_wall_format(-1, buf, sizeof buf));
    ET_STREQ(buf, "1969-12-31T23:59:59.999Z");
    ET_OK(engram_wall_format(253402300799999ll, buf, sizeof buf));
    ET_STREQ(buf, "9999-12-31T23:59:59.999Z");
    ET_OK(engram_wall_format(1790112000123ll, buf, sizeof buf));
    ET_STREQ(buf, "2026-09-22T21:20:00.123Z");
    ET_RC(engram_wall_format(0, buf, 24u), ENGRAM_E_ARG);
    ET_RC(engram_wall_format(0, NULL, 64u), ENGRAM_E_ARG);
    ET_RC(engram_wall_format(253402300800000ll, buf, sizeof buf), ENGRAM_E_ARG);   /* year 10000 */
}

static void test_files(void)
{
    uint8_t *data = NULL;
    size_t len = 0, i;
    uint64_t sz = 0;
    uint8_t bin[300];
    const char *path = SCRATCH_DIR "/atomic.bin";

    ET_SECTION("files: directory creation is idempotent");
    ET_OK(engram_dir_make(SCRATCH_DIR));
    ET_OK(engram_dir_make(SCRATCH_DIR));

    ET_SECTION("files: an absent file is NOT_FOUND, not IO");
    (void)engram_file_remove(path);
    ET_RC(engram_file_read(path, &data, &len), ENGRAM_E_NOTFOUND);
    ET_CHECK(data == NULL && len == 0u);
    ET_CHECK(!engram_file_exists(path));
    ET_RC(engram_file_size(path, &sz), ENGRAM_E_NOTFOUND);
    ET_RC(engram_file_remove(path), ENGRAM_E_NOTFOUND);

    ET_SECTION("files: binary round trip, including NUL bytes, with a terminator not counted");
    for (i = 0; i < sizeof bin; i++) bin[i] = (uint8_t)(i * 37u + 1u);
    bin[7] = 0; bin[150] = 0;
    ET_OK(engram_file_write_atomic(path, bin, sizeof bin));
    ET_CHECK(engram_file_exists(path));
    ET_OK(engram_file_size(path, &sz));
    ET_EQ_U64(sz, sizeof bin);
    ET_OK(engram_file_read(path, &data, &len));
    ET_EQ_U64(len, sizeof bin);
    ET_CHECK(data && memcmp(data, bin, sizeof bin) == 0);
    ET_CHECK(data && data[len] == 0);
    engram_free(data);

    ET_SECTION("files: an empty file round-trips to a valid empty buffer");
    ET_OK(engram_file_write_atomic(path, "", 0u));
    ET_OK(engram_file_read(path, &data, &len));
    ET_EQ_U64(len, 0u);
    ET_CHECK(data != NULL);
    engram_free(data);

    ET_SECTION("files: overwrite replaces the whole file");
    ET_OK(engram_file_write_atomic(path, "first version, longer", 21u));
    ET_OK(engram_file_write_atomic(path, "second", 6u));
    ET_OK(engram_file_read(path, &data, &len));
    ET_EQ_U64(len, 6u);
    ET_CHECK(data && memcmp(data, "second", 6u) == 0);
    engram_free(data);

    ET_SECTION("files: argument errors are refused");
    ET_RC(engram_file_write_atomic(NULL, "x", 1u), ENGRAM_E_ARG);
    ET_RC(engram_file_write_atomic(path, NULL, 1u), ENGRAM_E_ARG);
    ET_RC(engram_file_read(NULL, &data, &len), ENGRAM_E_ARG);
    ET_RC(engram_file_read(path, NULL, &len), ENGRAM_E_ARG);

    ET_OK(engram_file_remove(path));
}

/* ---- a directory scanner used to prove no temporary is left behind --------------------------- */
typedef struct { unsigned total, orphans; } dir_census;

static int census_cb(const char *name, void *user)
{
    dir_census *c = (dir_census *)user;
    c->total++;
    if (engram_tmp_name_is_orphan(name)) c->orphans++;
    return 0;
}

static unsigned count_orphans(const char *dir)
{
    dir_census c;
    memset(&c, 0, sizeof c);
    if (engram_dir_list(dir, census_cb, &c) != ENGRAM_OK) return 9999u;
    return c.orphans;
}

/* THE PROPERTY THE WHOLE STORAGE LAYER RESTS ON: fail every IO operation of an atomic write in turn.
 * After EVERY one the target must hold the complete OLD or the complete NEW contents -- never a torn
 * mixture -- and before the commit point it must still hold the OLD. And no temporary may remain. */
static void test_files_atomic_under_fault(void)
{
    const char *path = SCRATCH_DIR "/survivor.txt";
    const char *old_text = "the old memory, intact";
    const char *new_text = "the new memory, which must arrive whole or not at all";
    size_t lo = strlen(old_text), ln = strlen(new_text);
    uint8_t *data;
    size_t len;
    uint64_t k, ops;
    unsigned torn = 0, unexpected_ok = 0, refused_clean = 0, after_commit = 0, pre_commit_lost = 0;

    ET_SECTION("atomicity: count the IO operations one write performs");
    ET_OK(engram_file_write_atomic(path, old_text, lo));
    engram_io_reset();
    ET_OK(engram_file_write_atomic(path, old_text, lo));
    ops = engram_io_ops();
    ET_CHECKF(ops >= 4u, "an atomic write took only %llu IO ops", (unsigned long long)ops);
    printf("       one atomic write = %llu IO operations\n", (unsigned long long)ops);

    ET_SECTION("atomicity: fail each operation in turn -- never torn, old intact before the commit");
    for (k = 1; k <= ops; k++) {
        engram_rc rc;
        int is_old, is_new;
        ET_OK(engram_file_write_atomic(path, old_text, lo));         /* reset the target */
        engram_io_reset();
        engram_io_fail_at(k);
        rc = engram_file_write_atomic(path, new_text, ln);
        engram_io_fail_at(0u);
        if (engram_file_read(path, &data, &len) != ENGRAM_OK) { torn++; continue; }
        is_old = len == lo && memcmp(data, old_text, lo) == 0;
        is_new = len == ln && memcmp(data, new_text, ln) == 0;
        engram_free(data);
        if (!is_old && !is_new) torn++;
        if (rc == ENGRAM_OK)  unexpected_ok++;      /* an injected failure the write did not report */
        else if (is_new)      after_commit++;       /* failed AFTER the rename: durability unconfirmed */
        else                  refused_clean++;
        /* Every operation before the last precedes the rename, so the old file must be untouched. */
        if (k < ops && !is_old) pre_commit_lost++;
    }
    ET_EQ_U64(torn, 0u);
    ET_EQ_U64(unexpected_ok, 0u);
    ET_EQ_U64(pre_commit_lost, 0u);
    ET_CHECK(refused_clean >= ops - 1u);
    printf("       %u refused with the old file intact, %u after the commit point (new file whole), 0 torn\n",
           refused_clean, after_commit);

    ET_SECTION("atomicity: no failed write leaves a temporary behind");
    ET_EQ_U64(count_orphans(SCRATCH_DIR), 0u);
    ET_OK(engram_file_remove(path));
}

static void test_dirs_and_orphans(void)
{
    char *j;
    unsigned removed = 99u;
    dir_census c;
    const char *keep = SCRATCH_DIR "/notes.tmp.txt";

    ET_SECTION("path_join: adds a separator only when needed");
    j = engram_path_join("a/b", "c.txt");  ET_STREQ(j, "a/b/c.txt"); engram_free(j);
    j = engram_path_join("a/b/", "c.txt"); ET_STREQ(j, "a/b/c.txt"); engram_free(j);
    j = engram_path_join("", "c.txt");     ET_STREQ(j, "c.txt");     engram_free(j);
    ET_CHECK(engram_path_join(NULL, "x") == NULL);

    ET_SECTION("orphan matcher: strict -- only the atomic writer's own names");
    ET_CHECK( engram_tmp_name_is_orphan("store.bin.tmp.4242.7"));
    ET_CHECK( engram_tmp_name_is_orphan("a.tmp.1.2"));
    ET_CHECK(!engram_tmp_name_is_orphan(".tmp.1.2"));             /* empty base      */
    ET_CHECK(!engram_tmp_name_is_orphan("notes.tmp.txt"));        /* not digits      */
    ET_CHECK(!engram_tmp_name_is_orphan("a.tmp.12"));             /* one number only */
    ET_CHECK(!engram_tmp_name_is_orphan("a.tmp.1.2.bak"));        /* trailing suffix */
    ET_CHECK(!engram_tmp_name_is_orphan("a.tmp..2"));             /* empty field     */
    ET_CHECK(!engram_tmp_name_is_orphan("store.bin"));
    ET_CHECK(!engram_tmp_name_is_orphan(NULL));

    ET_SECTION("sweep: removes planted orphans, never touches a user file");
    ET_OK(engram_file_write_atomic(SCRATCH_DIR "/mem.bin.tmp.31337.1", "dead", 4u));
    ET_OK(engram_file_write_atomic(SCRATCH_DIR "/mem.bin.tmp.31337.2", "dead", 4u));
    ET_OK(engram_file_write_atomic(keep, "a user's file", 13u));
    ET_EQ_U64(count_orphans(SCRATCH_DIR), 2u);
    ET_OK(engram_tmp_sweep(SCRATCH_DIR, &removed));
    ET_EQ_U64(removed, 2u);
    ET_EQ_U64(count_orphans(SCRATCH_DIR), 0u);
    ET_CHECK(engram_file_exists(keep));
    ET_OK(engram_file_remove(keep));

    ET_SECTION("dir_list: counts regular files, skips . and .., refuses an absent dir");
    {
        unsigned before;
        memset(&c, 0, sizeof c);
        ET_OK(engram_dir_list(SCRATCH_DIR, census_cb, &c));
        before = c.total;                                 /* a delta, so leftovers cannot skew it */
        ET_OK(engram_file_write_atomic(SCRATCH_DIR "/one", "1", 1u));
        ET_OK(engram_file_write_atomic(SCRATCH_DIR "/two", "2", 1u));
        memset(&c, 0, sizeof c);
        ET_OK(engram_dir_list(SCRATCH_DIR, census_cb, &c));
        ET_EQ_U64(c.total, before + 2u);
    }
    ET_RC(engram_dir_list(SCRATCH_DIR "/does-not-exist", census_cb, &c), ENGRAM_E_NOTFOUND);
    ET_RC(engram_dir_list(NULL, census_cb, &c), ENGRAM_E_ARG);
    ET_OK(engram_file_remove(SCRATCH_DIR "/one"));
    ET_OK(engram_file_remove(SCRATCH_DIR "/two"));
}

static void test_entropy_host(void)
{
    uint8_t a[64], b[64];
    size_t i;
    int zero = 1;
    engram_idle idle;

    ET_SECTION("entropy: fills, differs between calls, is not all zero");
    memset(a, 0, sizeof a);
    memset(b, 0, sizeof b);
    ET_OK(engram_os_random(a, sizeof a));
    ET_OK(engram_os_random(b, sizeof b));
    for (i = 0; i < sizeof a; i++) if (a[i]) zero = 0;
    ET_CHECK(!zero);
    ET_CHECK(memcmp(a, b, sizeof a) != 0);
    ET_OK(engram_os_random(a, 0u));
    ET_RC(engram_os_random(NULL, 8u), ENGRAM_E_ARG);

    ET_SECTION("host: cpu count and the idle signal");
    ET_CHECK(engram_cpu_count() >= 1u);
    ET_OK(engram_idle_query(&idle));
    ET_RC(engram_idle_query(NULL), ENGRAM_E_ARG);
    printf("       cpus=%u idle.known=%d idle_ms=%llu mains=%d battery=%d\n",
           engram_cpu_count(), idle.known, (unsigned long long)idle.idle_ms,
           idle.on_mains, idle.battery_pct);
}

/* ==================================================================================================
 * LOG
 * ============================================================================================== */
typedef struct {
    char     last[ENGRAM_LOG_LINE + 8];
    unsigned calls;
    int      fail;
} cap_sink;

static int capture(engram_log_level level, const char *line, void *user)
{
    cap_sink *c = (cap_sink *)user;
    (void)level;
    c->calls++;
    strncpy(c->last, line, sizeof c->last - 1u);
    c->last[sizeof c->last - 1u] = 0;
    return c->fail ? -1 : 0;
}

static void test_log(void)
{
    cap_sink c;
    engram_log_stats s0, s1;
    char big[ENGRAM_LOG_LINE * 2];
    size_t i;

    memset(&c, 0, sizeof c);
    engram_log_set_sink(capture, &c);
    engram_log_set_level(ENGRAM_LOG_INFO);

    ET_SECTION("log: a line carries timestamp, level, module and message");
    engram_log(ENGRAM_LOG_INFO, "core", "consolidated %d episodes", 7);
    ET_EQ_U64(c.calls, 1u);
    ET_CHECK(strlen(c.last) > 24u && c.last[4] == '-' && c.last[10] == 'T' && c.last[23] == 'Z');
    ET_CHECK(strstr(c.last, " INFO  core: consolidated 7 episodes") != NULL);

    ET_SECTION("log: the level threshold filters, and counts what it filtered");
    engram_log_stats_get(&s0);
    engram_log(ENGRAM_LOG_DEBUG, "core", "hidden");
    engram_log_stats_get(&s1);
    ET_EQ_U64(c.calls, 1u);
    ET_EQ_U64(s1.filtered, s0.filtered + 1u);

    ET_SECTION("log: an embedded newline cannot forge a second entry");
    engram_log(ENGRAM_LOG_WARN, "core", "line one\n2026-01-01T00:00:00.000Z ERROR fake: injected");
    ET_CHECK(strchr(c.last, '\n') == NULL);
    ET_CHECK(strstr(c.last, "line one 2026") != NULL);

    ET_SECTION("log: an over-long line is cut, marked, and COUNTED");
    engram_log_stats_get(&s0);
    for (i = 0; i < sizeof big - 1u; i++) big[i] = (char)('a' + (char)(i % 26u));
    big[sizeof big - 1u] = 0;
    engram_log(ENGRAM_LOG_ERROR, "core", "%s", big);
    engram_log_stats_get(&s1);
    ET_EQ_U64(s1.truncated, s0.truncated + 1u);
    ET_CHECK(strlen(c.last) == ENGRAM_LOG_LINE - 1u);
    ET_CHECK(strcmp(c.last + strlen(c.last) - 3u, "...") == 0);
    ET_PLANT_LOG_LOSS(1u);

    ET_SECTION("log: a failing sink is counted, never silent");
    engram_log_stats_get(&s0);
    c.fail = 1;
    engram_log(ENGRAM_LOG_ERROR, "core", "this write fails");
    c.fail = 0;
    engram_log_stats_get(&s1);
    ET_EQ_U64(s1.write_failed, s0.write_failed + 1u);
    ET_PLANT_LOG_LOSS(1u);

    ET_SECTION("log: level names, and a garbage level promoted to ERROR");
    ET_STREQ(engram_log_level_name(ENGRAM_LOG_TRACE), "TRACE");
    ET_STREQ(engram_log_level_name((engram_log_level)99), "?????");
    engram_log((engram_log_level)99, "core", "garbage level");
    ET_CHECK(strstr(c.last, " ERROR core: garbage level") != NULL);

    engram_log_set_sink(NULL, NULL);
}

/* ==================================================================================================
 * SERIALISATION
 * ============================================================================================== */
static void test_buf_roundtrip(void)
{
    engram_wbuf w;
    engram_rbuf r;
    uint8_t *out = NULL;
    size_t n = 0, bl = 0;
    char *s;
    const uint8_t *blob;
    float fs[3] = { 1.5f, -0.0f, 3.25e-7f }, gs[3];

    ET_SECTION("buf: every type round-trips exactly");
    engram_wbuf_init(&w);
    engram_wbuf_u8(&w, 0xABu);
    engram_wbuf_u16(&w, 0xBEEFu);
    engram_wbuf_u32(&w, 0xDEADBEEFu);
    engram_wbuf_u64(&w, 0x0123456789ABCDEFull);
    engram_wbuf_i32(&w, -123456789);
    engram_wbuf_i64(&w, -1234567890123456789ll);
    engram_wbuf_f32(&w, -2.5f);
    engram_wbuf_f64(&w, 3.141592653589793);
    engram_wbuf_str(&w, "engram");
    engram_wbuf_blob(&w, "\x00\x01\x02", 3u);
    engram_wbuf_f32s(&w, fs, 3u);
    engram_wbuf_zeros(&w, 5u);
    ET_OK(engram_wbuf_finish(&w, &out, &n));

    engram_rbuf_init(&r, out, n);
    ET_EQ_U64(engram_rbuf_u8(&r), 0xABu);
    ET_EQ_U64(engram_rbuf_u16(&r), 0xBEEFu);
    ET_EQ_U64(engram_rbuf_u32(&r), 0xDEADBEEFu);
    ET_EQ_U64(engram_rbuf_u64(&r), 0x0123456789ABCDEFull);
    ET_CHECK(engram_rbuf_i32(&r) == -123456789);
    ET_CHECK(engram_rbuf_i64(&r) == -1234567890123456789ll);
    ET_CHECK(engram_rbuf_f32(&r) == -2.5f);
    ET_CHECK(engram_rbuf_f64(&r) == 3.141592653589793);
    s = engram_rbuf_str(&r, 64u);
    ET_STREQ(s, "engram");
    engram_free(s);
    blob = engram_rbuf_blob(&r, 16u, &bl);
    ET_CHECK(blob && bl == 3u && blob[0] == 0 && blob[2] == 2);
    ET_CHECK(engram_rbuf_f32s_finite(&r, gs, 3u));
    ET_CHECK(memcmp(fs, gs, sizeof fs) == 0);        /* bitwise: -0.0 must stay -0.0 */
    ET_CHECK(engram_rbuf_skip(&r, 5u));
    ET_OK(engram_rbuf_end(&r));
    engram_free(out);

    ET_SECTION("buf: little-endian on the wire regardless of host byte order");
    engram_wbuf_init(&w);
    engram_wbuf_u32(&w, 0x01020304u);
    ET_OK(engram_wbuf_finish(&w, &out, &n));
    ET_CHECK(n == 4u && out[0] == 0x04 && out[1] == 0x03 && out[2] == 0x02 && out[3] == 0x01);
    engram_free(out);

    ET_SECTION("buf: an empty writer finishes to a valid zero-length buffer");
    engram_wbuf_init(&w);
    ET_OK(engram_wbuf_finish(&w, &out, &n));
    ET_CHECK(out != NULL && n == 0u);
    engram_free(out);

    ET_SECTION("buf: patching an earlier length");
    engram_wbuf_init(&w);
    engram_wbuf_u32(&w, 0u);
    engram_wbuf_str(&w, "payload");
    engram_wbuf_put_u32_at(&w, 0u, (uint32_t)engram_wbuf_len(&w));
    ET_OK(engram_wbuf_status(&w));
    engram_wbuf_put_u32_at(&w, engram_wbuf_len(&w) - 2u, 1u);     /* straddles the end: a bug */
    ET_RC(engram_wbuf_status(&w), ENGRAM_E_INTERNAL);
    ET_RC(engram_wbuf_finish(&w, &out, &n), ENGRAM_E_INTERNAL);
    ET_CHECK(out == NULL && n == 0u);
}

static void test_buf_hostile(void)
{
    engram_wbuf w;
    engram_rbuf r;
    uint8_t *out = NULL;
    size_t n = 0;
    uint8_t five[5] = { 1, 2, 3, 4, 5 };
    uint8_t dst[8];
    uint8_t nan_bits[4], inf_bits[4];
    uint8_t lying_len[8], nul_str[7];
    char *s;
    float f[2];

    ET_SECTION("reader: a truncated field reads as ZERO, and the error is sticky");
    engram_rbuf_init(&r, five, sizeof five);
    ET_EQ_U64(engram_rbuf_u32(&r), 0x04030201u);
    ET_EQ_U64(engram_rbuf_u32(&r), 0u);              /* only 1 byte left: overrun */
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);
    ET_EQ_U64(engram_rbuf_u8(&r), 0u);               /* sticky: even a fitting read now fails */
    memset(dst, 0xEE, sizeof dst);
    ET_CHECK(!engram_rbuf_bytes(&r, dst, 4u));
    ET_CHECK(dst[0] == 0 && dst[3] == 0);            /* zeroed, never stale */

    ET_SECTION("reader: unconsumed trailing bytes fail the parse");
    engram_rbuf_init(&r, five, sizeof five);
    (void)engram_rbuf_u32(&r);
    ET_RC(engram_rbuf_end(&r), ENGRAM_E_FORMAT);

    ET_SECTION("reader: NaN and infinity are refused where a finite float is required");
    engram_le_put_u32(nan_bits, 0x7FC00000u);
    engram_le_put_u32(inf_bits, 0x7F800000u);
    engram_rbuf_init(&r, nan_bits, 4u);
    ET_CHECK(engram_rbuf_f32_finite(&r) == 0.0f);
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);
    engram_rbuf_init(&r, inf_bits, 4u);
    (void)engram_rbuf_f32_finite(&r);
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);
    {
        uint8_t two[8];
        engram_le_put_u32(two, 0x3F800000u);          /* 1.0 */
        engram_le_put_u32(two + 4, 0xFF800000u);      /* -inf */
        engram_rbuf_init(&r, two, 8u);
        f[0] = 9.0f; f[1] = 9.0f;
        ET_CHECK(!engram_rbuf_f32s_finite(&r, f, 2u));
        ET_CHECK(f[0] == 0.0f && f[1] == 0.0f);       /* never a partially-filled array */
    }

    ET_SECTION("reader: a length field that lies is refused, not trusted");
    engram_le_put_u32(lying_len, 0xFFFFFFF0u);
    memcpy(lying_len + 4, "abcd", 4u);
    engram_rbuf_init(&r, lying_len, sizeof lying_len);
    s = engram_rbuf_str(&r, 1u << 20);
    ET_CHECK(s == NULL);
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);
    engram_le_put_u32(lying_len, 4u);
    engram_rbuf_init(&r, lying_len, sizeof lying_len);
    ET_CHECK(engram_rbuf_str(&r, 3u) == NULL);       /* over the caller's maxlen */
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);

    ET_SECTION("reader: a string with an embedded NUL is refused");
    engram_le_put_u32(nul_str, 3u);
    nul_str[4] = 'a'; nul_str[5] = 0; nul_str[6] = 'b';
    engram_rbuf_init(&r, nul_str, sizeof nul_str);
    ET_CHECK(engram_rbuf_str(&r, 16u) == NULL);
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_FORMAT);

    ET_SECTION("reader: NULL with a length is refused at init");
    engram_rbuf_init(&r, NULL, 10u);
    ET_RC(engram_rbuf_status(&r), ENGRAM_E_ARG);

    ET_SECTION("writer: an allocation failure is sticky and surfaces at finish");
    engram_wbuf_init(&w);
    engram_alloc_fail_at(1u);
    engram_wbuf_u32(&w, 1u);                         /* fails */
    engram_wbuf_u32(&w, 2u);                         /* no-op */
    engram_wbuf_str(&w, "ignored");
    ET_RC(engram_wbuf_status(&w), ENGRAM_E_MEM);
    ET_RC(engram_wbuf_finish(&w, &out, &n), ENGRAM_E_MEM);
    ET_CHECK(out == NULL && n == 0u);

    ET_SECTION("writer: NULL source with a length is refused");
    engram_wbuf_init(&w);
    engram_wbuf_bytes(&w, NULL, 4u);
    ET_RC(engram_wbuf_finish(&w, &out, &n), ENGRAM_E_ARG);
}

/* ==================================================================================================
 * RNG
 * ============================================================================================== */
static void test_rng(void)
{
    engram_rng a, b;
    uint64_t v, seen[1000];
    uint32_t perm[100], counts[10];
    int i, j, ok, overlap;
    double sum = 0.0, sq = 0.0, x, lo = 99.0, hi = -99.0, chi = 0.0;

    ET_SECTION("rng: pinned to an independent xoshiro256** implementation");
    engram_rng_seed(&a, 0u);
    ET_EQ_U64(engram_rng_u64(&a), 0x99EC5F36CB75F2B4ull);
    ET_EQ_U64(engram_rng_u64(&a), 0xBF6E1F784956452Aull);
    ET_EQ_U64(engram_rng_u64(&a), 0x1A5F849D4933E6E0ull);
    engram_rng_seed(&a, 12345u);
    ET_EQ_U64(engram_rng_u64(&a), 0xBE6A36374160D49Bull);
    ET_EQ_U64(engram_rng_u64(&a), 0x214AAA0637A688C6ull);
    ET_EQ_U64(engram_rng_u64(&a), 0xF69D16DE9954D388ull);

    ET_SECTION("rng: the same seed replays the same stream");
    engram_rng_seed(&a, 777u);
    engram_rng_seed(&b, 777u);
    for (ok = 1, i = 0; i < 10000; i++) if (engram_rng_u64(&a) != engram_rng_u64(&b)) ok = 0;
    ET_CHECK(ok);

    ET_SECTION("rng: below(n) stays in range, is unbiased, and below(0) is 0");
    engram_rng_seed(&a, 1u);
    ET_EQ_U64(engram_rng_below(&a, 0u), 0u);
    for (ok = 1, i = 0; i < 100000; i++) if (engram_rng_below(&a, 7u) >= 7u) ok = 0;
    ET_CHECK(ok);
    memset(counts, 0, sizeof counts);
    for (i = 0; i < 100000; i++) counts[engram_rng_below(&a, 10u)]++;
    for (i = 0; i < 10; i++) {
        double d = (double)counts[i] - 10000.0;
        chi += d * d / 10000.0;
    }
    /* 9 degrees of freedom: chi-square above 27.88 happens by chance 0.1% of the time. */
    ET_CHECKF(chi < 27.88, "chi-square %.2f over 10 bins", chi);

    ET_SECTION("rng: unit() is in [0, 1)");
    for (ok = 1, i = 0; i < 100000; i++) {
        x = engram_rng_unit(&a);
        if (!(x >= 0.0 && x < 1.0)) ok = 0;
    }
    ET_CHECK(ok);
    for (ok = 1, i = 0; i < 100000; i++) {
        float y = engram_rng_unitf(&a);
        if (!(y >= 0.0f && y < 1.0f)) ok = 0;
    }
    ET_CHECK(ok);

    ET_SECTION("rng: Irwin-Hall normal has mean 0, variance 1, and support [-6, 6]");
    for (i = 0; i < 200000; i++) {
        x = engram_rng_normal(&a);
        sum += x; sq += x * x;
        if (x < lo) lo = x;
        if (x > hi) hi = x;
    }
    x = sum / 200000.0;
    ET_CHECKF(x > -0.01 && x < 0.01, "mean %.5f", x);
    x = sq / 200000.0 - (sum / 200000.0) * (sum / 200000.0);
    ET_CHECKF(x > 0.98 && x < 1.02, "variance %.5f", x);
    ET_CHECK(lo >= -6.0 && hi <= 6.0);

    ET_SECTION("rng: shuffle produces a permutation");
    for (i = 0; i < 100; i++) perm[i] = (uint32_t)i;
    engram_rng_shuffle_u32(&a, perm, 100u);
    for (ok = 1, i = 0; i < 100; i++) {
        int found = 0;
        for (j = 0; j < 100; j++) if (perm[j] == (uint32_t)i) found++;
        if (found != 1) ok = 0;
    }
    ET_CHECK(ok);

    ET_SECTION("rng: jump yields a disjoint stream from the same seed");
    engram_rng_seed(&a, 42u);
    b = a;
    engram_rng_jump(&b);
    for (i = 0; i < 1000; i++) seen[i] = engram_rng_u64(&a);
    for (overlap = 0, i = 0; i < 1000; i++) {
        v = engram_rng_u64(&b);
        for (j = 0; j < 1000; j++) if (seen[j] == v) overlap++;
    }
    ET_CHECK(overlap == 0);
}

/* ==================================================================================================
 * CONCURRENCY
 * ==================================================================================================
 * Eight threads allocate, fill, verify and free at random, and log, all at once. Each thread writes a
 * byte pattern unique to itself and its slot, and checks it before freeing -- so if the allocator ever
 * handed one thread memory another thread still owned, it shows up as a WRONG BYTE, which is a much
 * earlier and more specific signal than an eventual crash. Run under ThreadSanitizer by `make tsan`. */
#define ET_THREADS 8u
#define ET_SLOTS   32u

typedef struct {
    unsigned id;
    unsigned iters;
    unsigned bad;
    unsigned logged;
} worker_arg;

static unsigned long g_count_sink_lines;          /* written only under the log's own lock */

static int count_sink(engram_log_level level, const char *line, void *user)
{
    (void)level; (void)line; (void)user;
    g_count_sink_lines++;
    return 0;
}

static int slot_intact(const unsigned char *b, size_t n, unsigned char want)
{
    size_t i;
    for (i = 0; i < n; i++) if (b[i] != want) return 0;
    return 1;
}

static void alloc_worker(void *p)
{
    worker_arg *w = (worker_arg *)p;
    engram_rng r;
    unsigned char *held[ET_SLOTS];
    size_t sz[ET_SLOTS];
    unsigned i, k;

    engram_rng_seed(&r, 1000u + w->id);
    memset(held, 0, sizeof held);
    memset(sz, 0, sizeof sz);
    for (i = 0; i < w->iters; i++) {
        unsigned char pat;
        k = (unsigned)engram_rng_below(&r, ET_SLOTS);
        pat = (unsigned char)((w->id * 31u + k * 7u + 1u) & 0xFFu);
        if (held[k]) {
            if (!slot_intact(held[k], sz[k], pat)) w->bad++;
            if (engram_alloc_check(held[k]) != ENGRAM_OK) w->bad++;
            engram_free(held[k]);
            held[k] = NULL;
        } else {
            size_t n = 1u + (size_t)engram_rng_below(&r, 512u);
            unsigned char *b = (unsigned char *)engram_malloc(n);
            if (!b) { w->bad++; continue; }
            memset(b, pat, n);
            held[k] = b;
            sz[k] = n;
        }
        if ((i & 1023u) == 0u) {
            engram_log(ENGRAM_LOG_INFO, "thread", "worker %u at iteration %u", w->id, i);
            w->logged++;
        }
    }
    for (k = 0; k < ET_SLOTS; k++) {
        if (!held[k]) continue;
        if (!slot_intact(held[k], sz[k],
                         (unsigned char)((w->id * 31u + k * 7u + 1u) & 0xFFu))) w->bad++;
        engram_free(held[k]);
    }
}

static void test_threads(void)
{
    engram_thread *t[ET_THREADS];
    worker_arg a[ET_THREADS];
    engram_alloc_stats s0, s1;
    uint64_t corrupt0;
    unsigned i, bad = 0, logged = 0, started = 0;

    ET_SECTION("threads: start and join");
    ET_RC(engram_thread_start(NULL, alloc_worker, NULL), ENGRAM_E_ARG);
    ET_RC(engram_thread_start(&t[0], NULL, NULL), ENGRAM_E_ARG);
    ET_RC(engram_thread_join(NULL), ENGRAM_E_ARG);

    ET_SECTION("threads: 8 workers x 20000 ops -- no foreign bytes, no leak, no corruption");
    engram_log_set_sink(count_sink, NULL);
    g_count_sink_lines = 0;
    engram_alloc_stats_get(&s0);
    corrupt0 = engram_alloc_corruption();
    for (i = 0; i < ET_THREADS; i++) {
        a[i].id = i; a[i].iters = 20000u; a[i].bad = 0; a[i].logged = 0;
        if (engram_thread_start(&t[i], alloc_worker, &a[i]) == ENGRAM_OK) started++;
        else t[i] = NULL;
    }
    ET_EQ_U64(started, ET_THREADS);
    for (i = 0; i < ET_THREADS; i++) if (t[i]) ET_OK(engram_thread_join(t[i]));
    for (i = 0; i < ET_THREADS; i++) { bad += a[i].bad; logged += a[i].logged; }
    engram_log_set_sink(NULL, NULL);
    engram_alloc_stats_get(&s1);

    ET_EQ_U64(bad, 0u);
    ET_EQ_U64(engram_alloc_corruption(), corrupt0);
    ET_EQ_U64(s1.live_blocks, s0.live_blocks);
    ET_EQ_U64(s1.live_bytes, s0.live_bytes);
    ET_EQ_U64(g_count_sink_lines, logged);            /* every line from every thread arrived */
    printf("       %u threads, %llu allocs, %u log lines, 0 foreign bytes\n", ET_THREADS,
           (unsigned long long)(s1.n_alloc - s0.n_alloc), logged);
}

int main(void)
{
    printf("ENGRAM P1.1 -- core runtime\n");
    ET_SELFTEST();
    /* A clean slate: a previous run killed mid-write may have left orphans that would skew counts. */
    if (engram_dir_make(SCRATCH_DIR) != ENGRAM_OK ||
        engram_tmp_sweep(SCRATCH_DIR, NULL) != ENGRAM_OK) {
        printf("cannot prepare %s\n", SCRATCH_DIR);
        return 2;
    }
    test_rc();
    test_hash();
    test_build_info();
    test_alloc_basic();
    test_alloc_grow();
    test_alloc_faults();
    test_alloc_detectors();
    test_clock();
    test_files();
    test_files_atomic_under_fault();
    test_dirs_and_orphans();
    test_entropy_host();
    test_log();
    test_buf_roundtrip();
    test_buf_hostile();
    test_rng();
    test_threads();
#ifdef ENGRAM_GATE_PLANT
    /* Compiled in ONLY by tools/gate.sh step 0, which requires this binary to FAIL -- the proof that a
     * failing check reaches the gate's verdict. The marker line proves the plant itself ran. */
    ET_SECTION("GATE PLANT: a deliberately failing check");
    printf("       ENGRAM-GATE-PLANT-EXECUTED\n");
    ET_CHECK(1 + 1 == 3);
#endif
    return et_report("test_core");
}
