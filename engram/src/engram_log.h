/* ==================================================================================================
 * engram_log.h -- one log, one writer, and every loss of a log line counted.
 * ==================================================================================================
 *
 * WHY LOGGING NEEDS DISCIPLINE AT ALL
 * ==================================================================================================
 * ENGRAM consolidates while nobody is watching. The log is how a person finds out, afterwards, what it
 * decided and why -- so a log line that was silently truncated, or silently dropped because the sink
 * failed, is a decision that happened with no record. Those losses are counted here, and the test
 * suite asserts both counters read zero.
 *
 * ONE WRITER. Every line goes through one lock to one sink. Two writers interleaving into one file
 * produce lines that are individually well-formed and collectively a lie about ordering, and nothing
 * downstream can detect it.
 *
 * WHY engram_log RETURNS NOTHING, stated because rule R1 says nothing that can fail is void:
 * logging is the one call made from inside error paths, and an error path that must then check
 * whether its own log line succeeded has nowhere to report THAT failure either. So logging defers its
 * report, exactly like the sticky buffers: every failure is recorded in engram_log_stats, which is
 * observable and asserted on. The failure is postponed, never lost.
 * ============================================================================================== */
#ifndef ENGRAM_LOG_H
#define ENGRAM_LOG_H

#include "engram.h"

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__MINGW32__) && defined(__MINGW_PRINTF_FORMAT)
#  define ENGRAM_PRINTF(f, a) __attribute__((format(__MINGW_PRINTF_FORMAT, f, a)))
#elif defined(__GNUC__) || defined(__clang__)
#  define ENGRAM_PRINTF(f, a) __attribute__((format(printf, f, a)))
#else
#  define ENGRAM_PRINTF(f, a)
#endif

typedef enum {
    ENGRAM_LOG_ERROR = 0,
    ENGRAM_LOG_WARN  = 1,
    ENGRAM_LOG_INFO  = 2,
    ENGRAM_LOG_DEBUG = 3,
    ENGRAM_LOG_TRACE = 4
} engram_log_level;

#define ENGRAM_LOG_LEVELS 5u

/* The longest line the log will emit, including its timestamp and level prefix. Longer messages are
 * cut, marked with a trailing "...", and COUNTED -- never silently shortened. */
#ifndef ENGRAM_LOG_LINE
#define ENGRAM_LOG_LINE 2048u
#endif

/* A sink receives one complete, NUL-terminated line WITHOUT a trailing newline, and returns 0 on
 * success or nonzero if it could not record it (counted as write_failed). */
typedef int (*engram_log_sink)(engram_log_level level, const char *line, void *user);

/* Lines above the threshold are discarded before formatting, and counted. Default: INFO. */
void             engram_log_set_level(engram_log_level level);
engram_log_level engram_log_get_level(void);

/* Install a sink. NULL restores the default, which writes to stderr. */
void engram_log_set_sink(engram_log_sink sink, void *user);

void engram_log(engram_log_level level, const char *module, const char *fmt, ...) ENGRAM_PRINTF(3, 4);
void engram_logv(engram_log_level level, const char *module, const char *fmt, va_list ap);

/* "ERROR", "WARN", ... ; never NULL. */
const char *engram_log_level_name(engram_log_level level);

typedef struct {
    uint64_t emitted[ENGRAM_LOG_LEVELS];  /* lines delivered to the sink, per level              */
    uint64_t filtered;                    /* discarded by the level threshold -- intentional     */
    uint64_t truncated;                   /* cut to fit ENGRAM_LOG_LINE -- must be 0             */
    uint64_t write_failed;                /* the sink reported failure -- must be 0              */
    uint64_t format_failed;               /* vsnprintf itself failed -- must be 0                */
} engram_log_stats;

void engram_log_stats_get(engram_log_stats *out);
void engram_log_stats_reset(void);

/* The sum of the three "must be 0" counters. */
uint64_t engram_log_losses(void);

#define ENGRAM_LOGE(mod, ...) engram_log(ENGRAM_LOG_ERROR, (mod), __VA_ARGS__)
#define ENGRAM_LOGW(mod, ...) engram_log(ENGRAM_LOG_WARN,  (mod), __VA_ARGS__)
#define ENGRAM_LOGI(mod, ...) engram_log(ENGRAM_LOG_INFO,  (mod), __VA_ARGS__)
#define ENGRAM_LOGD(mod, ...) engram_log(ENGRAM_LOG_DEBUG, (mod), __VA_ARGS__)
#define ENGRAM_LOGT(mod, ...) engram_log(ENGRAM_LOG_TRACE, (mod), __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_LOG_H */
