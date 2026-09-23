/* ==================================================================================================
 * engram_core.c -- return codes, the deterministic mix, and what this binary is.
 * ============================================================================================== */
#include "engram.h"

#include <string.h>

/* ---- RETURN CODES ------------------------------------------------------------------------------
 * A table rather than a switch so the two lookups (message and symbolic name) cannot drift apart:
 * one row per code, both strings side by side, and a code added without a row is caught by the
 * P1.1 test that walks every value from ENGRAM_E_INTERNAL to ENGRAM_OK. */
typedef struct {
    engram_rc   rc;
    const char *name;
    const char *msg;
} engram_rc_row;

static const engram_rc_row ENGRAM_RC_TABLE[] = {
    { ENGRAM_OK,         "ENGRAM_OK",         "success" },
    { ENGRAM_E_ARG,      "ENGRAM_E_ARG",      "invalid argument" },
    { ENGRAM_E_MEM,      "ENGRAM_E_MEM",      "out of memory" },
    { ENGRAM_E_IO,       "ENGRAM_E_IO",       "input/output error" },
    { ENGRAM_E_FORMAT,   "ENGRAM_E_FORMAT",   "malformed data" },
    { ENGRAM_E_VERSION,  "ENGRAM_E_VERSION",  "unsupported version" },
    { ENGRAM_E_EMPTY,    "ENGRAM_E_EMPTY",    "store is empty" },
    { ENGRAM_E_SHORT,    "ENGRAM_E_SHORT",    "input too short" },
    { ENGRAM_E_OVERFLOW, "ENGRAM_E_OVERFLOW", "size computation would overflow" },
    { ENGRAM_E_AUTH,     "ENGRAM_E_AUTH",     "integrity or authentication check failed" },
    { ENGRAM_E_NOTFOUND, "ENGRAM_E_NOTFOUND", "not found" },
    { ENGRAM_E_FULL,     "ENGRAM_E_FULL",     "at capacity" },
    { ENGRAM_E_STATE,    "ENGRAM_E_STATE",    "operation not valid in this state" },
    { ENGRAM_E_UTF8,     "ENGRAM_E_UTF8",     "invalid UTF-8" },
    { ENGRAM_E_INTERNAL, "ENGRAM_E_INTERNAL", "internal invariant violated" }
};

static const engram_rc_row *engram_rc_find(engram_rc rc)
{
    size_t i;
    for (i = 0; i < sizeof ENGRAM_RC_TABLE / sizeof ENGRAM_RC_TABLE[0]; i++)
        if (ENGRAM_RC_TABLE[i].rc == rc) return &ENGRAM_RC_TABLE[i];
    return NULL;
}

const char *engram_strerror(engram_rc rc)
{
    const engram_rc_row *r = engram_rc_find(rc);
    return r ? r->msg : "unknown error code";
}

const char *engram_rcname(engram_rc rc)
{
    const engram_rc_row *r = engram_rc_find(rc);
    return r ? r->name : "ENGRAM_E_UNKNOWN";
}

/* ---- THE MIX -----------------------------------------------------------------------------------
 * engram_mix64 and engram_mix2 are defined inline in engram.h. Every constant in them is load-bearing
 * across the whole tree: the encoder's dimensions and signs and the episodic projection's matrix all
 * come out of them, and none of those are stored. Changing one digit reads every existing store
 * through a different geometry. test_core.c pins known outputs so that cannot happen silently. */

/* FNV-1a, 64-bit, then finalised. FNV alone has weak low bits for short keys, and a caller that takes
 * `hash mod n` uses exactly those. The finaliser fixes that without changing what FNV is good at,
 * which is being cheap and byte-at-a-time. */
uint64_t engram_hash_bytes(const void *p, size_t n, uint64_t seed)
{
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 0xCBF29CE484222325ull ^ engram_mix64(seed);
    size_t i;
    for (i = 0; i < n; i++) {
        h ^= (uint64_t)b[i];
        h *= 0x100000001B3ull;
    }
    return engram_mix64(h ^ (uint64_t)n);
}

/* ---- WHAT THIS BINARY IS ----------------------------------------------------------------------- */
#if defined(_WIN64)
#  define ENGRAM_PLATFORM "windows-x86_64"
#elif defined(_WIN32)
#  define ENGRAM_PLATFORM "windows-x86"
#elif defined(__APPLE__) && defined(__aarch64__)
#  define ENGRAM_PLATFORM "macos-arm64"
#elif defined(__APPLE__)
#  define ENGRAM_PLATFORM "macos-x86_64"
#elif defined(__linux__) && defined(__aarch64__)
#  define ENGRAM_PLATFORM "linux-arm64"
#elif defined(__linux__) && defined(__x86_64__)
#  define ENGRAM_PLATFORM "linux-x86_64"
#elif defined(__linux__)
#  define ENGRAM_PLATFORM "linux"
#else
#  define ENGRAM_PLATFORM "unknown"
#endif

#if defined(__clang__)
#  define ENGRAM_COMPILER "clang " __clang_version__
#elif defined(__MINGW64__)
#  define ENGRAM_COMPILER "mingw-w64 gcc " __VERSION__
#elif defined(__GNUC__)
#  define ENGRAM_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#  define ENGRAM_COMPILER "msvc"
#else
#  define ENGRAM_COMPILER "unknown"
#endif

engram_rc engram_build_info_get(engram_build_info *out)
{
    if (!out) return ENGRAM_E_ARG;
    memset(out, 0, sizeof *out);
    out->major     = ENGRAM_VER_MAJOR;
    out->minor     = ENGRAM_VER_MINOR;
    out->patch     = ENGRAM_VER_PATCH;
    out->d         = ENGRAM_D;
    out->epi_tail  = ENGRAM_EPI_TAIL;
    out->string    = ENGRAM_VER_STRING;
    out->platform  = ENGRAM_PLATFORM;
    out->compiler  = ENGRAM_COMPILER;
    return ENGRAM_OK;
}
