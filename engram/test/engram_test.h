/* ==================================================================================================
 * engram_test.h -- the test harness every suite in this tree runs under.
 * ==================================================================================================
 *
 * WHAT EVERY SUITE PROVES BEFORE IT MAY REPORT SUCCESS
 * ==================================================================================================
 * A suite passes only if every check passed AND, at the end:
 *
 *   - the heap is clean: no live block the suite did not account for (a leak fails the suite)
 *   - the allocator saw no corruption the suite did not deliberately plant
 *   - the log lost no line (nothing truncated, no sink write failed)
 *
 * PLANTED DEFECTS ARE ACCOUNTED, NOT EXEMPTED. Some checks exist to prove a detector fires: overrun a
 * block and require the overrun counter to move. That leaves a counted corruption and a deliberately
 * leaked block behind. Weakening the end-of-suite invariant to tolerate them would also tolerate an
 * UNPLANTED corruption -- the one this whole apparatus exists to catch. So each planted defect is
 * declared with ET_PLANT_*, and the invariant requires the counters to equal EXACTLY what was planted.
 * One more, and the suite fails.
 *
 * AN INSTRUMENT THAT CANNOT FAIL IS NOT AN INSTRUMENT (R6). ET_SELFTEST() plants a failing check and
 * requires the harness to have counted it -- so a harness that silently stopped counting failures
 * cannot report a green run.
 * ============================================================================================== */
#ifndef ENGRAM_TEST_H
#define ENGRAM_TEST_H

#include <stdio.h>
#include <string.h>

#include "../src/engram.h"
#include "../src/engram_alloc.h"
#include "../src/engram_log.h"

static unsigned long et_pass;
static unsigned long et_fail;
static unsigned long et_sections;
static const char   *et_section = "";
static uint64_t      et_planted_corruption;
static uint64_t      et_planted_leak_blocks;
static uint64_t      et_planted_leak_bytes;
static uint64_t      et_planted_log_loss;

static int et_check_(int ok, const char *expr, const char *file, int line)
{
    if (ok) { et_pass++; return 1; }
    et_fail++;
    printf("    FAIL [%s] %s:%d: %s\n", et_section, file, line, expr);
    fflush(stdout);
    return 0;
}

#define ET_SECTION(name) \
    do { et_section = (name); et_sections++; printf("  -- %s\n", et_section); fflush(stdout); } while (0)

#define ET_CHECK(cond) ((void)et_check_((cond) ? 1 : 0, #cond, __FILE__, __LINE__))

#define ET_CHECKF(cond, ...)                                                                        \
    do {                                                                                            \
        if (!et_check_((cond) ? 1 : 0, #cond, __FILE__, __LINE__)) {                                \
            printf("         "); printf(__VA_ARGS__); printf("\n"); fflush(stdout);                 \
        }                                                                                           \
    } while (0)

#define ET_EQ_U64(a, b)                                                                             \
    do {                                                                                            \
        unsigned long long et_a_ = (unsigned long long)(a), et_b_ = (unsigned long long)(b);        \
        if (!et_check_(et_a_ == et_b_, #a " == " #b, __FILE__, __LINE__))                           \
            printf("         got 0x%016llX (%llu), want 0x%016llX (%llu)\n", et_a_, et_a_, et_b_, et_b_); \
    } while (0)

#define ET_RC(expr, want)                                                                           \
    do {                                                                                            \
        engram_rc et_got_ = (expr);                                                                 \
        if (!et_check_(et_got_ == (want), #expr " -> " #want, __FILE__, __LINE__))                  \
            printf("         got %s (%d)\n", engram_rcname(et_got_), (int)et_got_);                 \
    } while (0)

#define ET_OK(expr) ET_RC((expr), ENGRAM_OK)

#define ET_STREQ(a, b)                                                                              \
    do {                                                                                            \
        const char *et_sa_ = (a), *et_sb_ = (b);                                                    \
        int et_ok_ = et_sa_ && et_sb_ && strcmp(et_sa_, et_sb_) == 0;                               \
        if (!et_check_(et_ok_, #a " == " #b, __FILE__, __LINE__))                                   \
            printf("         got \"%s\", want \"%s\"\n", et_sa_ ? et_sa_ : "(null)",               \
                   et_sb_ ? et_sb_ : "(null)");                                                     \
    } while (0)

/* Declare a defect planted on purpose. */
#define ET_PLANT_CORRUPTION(n)      (et_planted_corruption += (uint64_t)(n))
#define ET_PLANT_LOG_LOSS(n)        (et_planted_log_loss += (uint64_t)(n))
#define ET_PLANT_LEAK(blocks, bytes) \
    (et_planted_leak_blocks += (uint64_t)(blocks), et_planted_leak_bytes += (uint64_t)(bytes))

/* R6: prove the harness counts a failure. The planted failure is then removed from the tally, so it
 * does not fail the suite -- but only AFTER it has been seen to register. */
#define ET_SELFTEST()                                                                               \
    do {                                                                                            \
        unsigned long et_before_ = et_fail;                                                         \
        printf("  -- harness self-test (one planted failure follows, and is expected)\n");          \
        (void)et_check_(0, "planted failure", __FILE__, __LINE__);                                  \
        if (et_fail != et_before_ + 1u) {                                                           \
            printf("  HARNESS BROKEN: a planted failure was not counted\n");                        \
            return 2;                                                                               \
        }                                                                                           \
        et_fail = et_before_;                                                                       \
    } while (0)

/* The end-of-suite verdict. Returns the process exit code. */
static int et_report(const char *suite)
{
    engram_alloc_stats st;
    engram_log_stats   ls;
    unsigned long long corrupt;

    et_section = "end-of-suite invariants";
    engram_alloc_quarantine_flush();
    engram_alloc_stats_get(&st);
    engram_log_stats_get(&ls);
    corrupt = (unsigned long long)engram_alloc_corruption();

    ET_CHECKF(corrupt == et_planted_corruption,
              "heap corruption %llu, planted %llu (double=%llu foreign=%llu overrun=%llu)",
              corrupt, (unsigned long long)et_planted_corruption,
              (unsigned long long)st.n_double_free, (unsigned long long)st.n_foreign_free,
              (unsigned long long)st.n_overrun);
    ET_CHECKF(st.live_blocks == et_planted_leak_blocks,
              "live blocks %llu (%llu bytes), planted leaks %llu (%llu bytes)",
              (unsigned long long)st.live_blocks, (unsigned long long)st.live_bytes,
              (unsigned long long)et_planted_leak_blocks, (unsigned long long)et_planted_leak_bytes);
    ET_CHECKF(st.live_bytes == et_planted_leak_bytes,
              "live bytes %llu, planted %llu",
              (unsigned long long)st.live_bytes, (unsigned long long)et_planted_leak_bytes);
    ET_CHECKF(engram_log_losses() == et_planted_log_loss,
              "log losses %llu, planted %llu (truncated=%llu write_failed=%llu format_failed=%llu)",
              (unsigned long long)engram_log_losses(), (unsigned long long)et_planted_log_loss,
              (unsigned long long)ls.truncated, (unsigned long long)ls.write_failed,
              (unsigned long long)ls.format_failed);

    printf("%s: %lu passed, %lu failed, %lu sections | allocs=%llu frees=%llu peak=%llu B\n",
           suite, et_pass, et_fail, et_sections,
           (unsigned long long)st.n_alloc, (unsigned long long)st.n_free,
           (unsigned long long)st.peak_bytes);
    printf("%s\n", et_fail ? "RESULT: FAIL" : "RESULT: PASS");
    fflush(stdout);
    return et_fail ? 1 : 0;
}

#endif /* ENGRAM_TEST_H */
