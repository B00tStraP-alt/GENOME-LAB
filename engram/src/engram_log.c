/* ==================================================================================================
 * engram_log.c -- one lock, one sink, and every lost line counted.
 * ==================================================================================================
 * Formatting happens OUTSIDE the lock (it is the slow part and touches only a stack buffer); delivery
 * happens INSIDE it (so lines reach the sink whole and in order). A sink is therefore called with the
 * log lock held, and must not itself call engram_log -- the lock is not recursive, and a sink that
 * logs would deadlock rather than interleave, which is the safer of the two failures.
 * ============================================================================================== */
#include "engram_log.h"
#include "engram_plat.h"

#include <stdio.h>
#include <string.h>

static engram_mutex     g_lock  = ENGRAM_MUTEX_INIT;
static engram_log_level g_level = ENGRAM_LOG_INFO;
static engram_log_sink  g_sink  = NULL;
static void            *g_user  = NULL;
static engram_log_stats g_st;

static const char *const ENGRAM_LOG_NAMES[ENGRAM_LOG_LEVELS] = {
    "ERROR", "WARN", "INFO", "DEBUG", "TRACE"
};

const char *engram_log_level_name(engram_log_level level)
{
    return (unsigned)level < ENGRAM_LOG_LEVELS ? ENGRAM_LOG_NAMES[level] : "?????";
}

static int engram_log_default_sink(engram_log_level level, const char *line, void *user)
{
    (void)level;
    (void)user;
    if (fputs(line, stderr) == EOF) return -1;
    if (fputc('\n', stderr) == EOF) return -1;
    return fflush(stderr) == 0 ? 0 : -1;
}

void engram_log_set_level(engram_log_level level)
{
    if ((unsigned)level >= ENGRAM_LOG_LEVELS) level = ENGRAM_LOG_TRACE;
    engram_mutex_lock(&g_lock);
    g_level = level;
    engram_mutex_unlock(&g_lock);
}

engram_log_level engram_log_get_level(void)
{
    engram_log_level v;
    engram_mutex_lock(&g_lock); v = g_level; engram_mutex_unlock(&g_lock);
    return v;
}

void engram_log_set_sink(engram_log_sink sink, void *user)
{
    engram_mutex_lock(&g_lock);
    g_sink = sink;
    g_user = sink ? user : NULL;
    engram_mutex_unlock(&g_lock);
}

/* One entry is one line. A message containing a newline would otherwise become two lines, the second
 * without a timestamp or level -- unparseable, and indistinguishable from an injected entry. Line
 * breaks become spaces; other C0 control characters become '?'. Tab is kept. */
static void engram_log_sanitise(char *s)
{
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '\n' || c == '\r')             *s = ' ';
        else if (c < 0x20u && c != '\t')        *s = '?';
        else if (c == 0x7Fu)                    *s = '?';
    }
}

void engram_logv(engram_log_level level, const char *module, const char *fmt, va_list ap)
{
    char line[ENGRAM_LOG_LINE];
    char ts[32];
    size_t pre = 0;
    int n, trunc = 0, ffail = 0, rc;
    engram_log_sink sink;
    void *user;

    /* An out-of-range level is promoted to ERROR rather than dropped: a caller passing garbage is
     * itself something that must appear in the log. */
    if ((unsigned)level >= ENGRAM_LOG_LEVELS) level = ENGRAM_LOG_ERROR;

    engram_mutex_lock(&g_lock);
    if (level > g_level) {
        g_st.filtered++;
        engram_mutex_unlock(&g_lock);
        return;
    }
    engram_mutex_unlock(&g_lock);

    /* The fallback must not look like a real time (that would be a fabricated record), and must not
     * contain "??" -- under -std=c99 trigraphs are live and "??-" silently becomes '~'. */
    if (engram_wall_format(engram_wall_ms(), ts, sizeof ts) != ENGRAM_OK)
        memcpy(ts, "XXXX-XX-XXTXX:XX:XX.XXXZ", 25u);

    n = snprintf(line, sizeof line, "%s %-5s %s: ", ts, engram_log_level_name(level),
                 module ? module : "-");
    if (n < 0) {
        ffail = 1;
        line[0] = 0;
    } else if ((size_t)n >= sizeof line) {
        trunc = 1;
    } else {
        pre = (size_t)n;
        n = vsnprintf(line + pre, sizeof line - pre, fmt ? fmt : "", ap);
        if (n < 0) {
            ffail = 1;
            line[pre] = 0;
        } else if ((size_t)n >= sizeof line - pre) {
            trunc = 1;
        }
    }
    if (trunc) {
        /* Mark the cut in the line itself, so a reader of the file knows without the counter. */
        size_t e = sizeof line - 1u;
        line[e] = 0;
        line[e - 1u] = '.'; line[e - 2u] = '.'; line[e - 3u] = '.';
    }
    engram_log_sanitise(line);

    engram_mutex_lock(&g_lock);
    sink = g_sink ? g_sink : engram_log_default_sink;
    user = g_user;
    rc = sink(level, line, user);
    if (rc != 0) g_st.write_failed++;
    else         g_st.emitted[level]++;
    if (trunc)   g_st.truncated++;
    if (ffail)   g_st.format_failed++;
    engram_mutex_unlock(&g_lock);
}

void engram_log(engram_log_level level, const char *module, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    engram_logv(level, module, fmt, ap);
    va_end(ap);
}

void engram_log_stats_get(engram_log_stats *out)
{
    if (!out) return;
    engram_mutex_lock(&g_lock);
    *out = g_st;
    engram_mutex_unlock(&g_lock);
}

void engram_log_stats_reset(void)
{
    engram_mutex_lock(&g_lock);
    memset(&g_st, 0, sizeof g_st);
    engram_mutex_unlock(&g_lock);
}

uint64_t engram_log_losses(void)
{
    uint64_t v;
    engram_mutex_lock(&g_lock);
    v = g_st.truncated + g_st.write_failed + g_st.format_failed;
    engram_mutex_unlock(&g_lock);
    return v;
}
