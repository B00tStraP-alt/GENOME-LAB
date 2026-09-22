/* ==================================================================================================
 * engram_plat.c -- the operating system, POSIX and Win32.
 * ============================================================================================== */
#ifndef _WIN32
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200809L
#  endif
#  ifndef _DEFAULT_SOURCE
#    define _DEFAULT_SOURCE 1
#  endif
#endif

#include "engram_plat.h"
#include "engram_alloc.h"
#include "engram_log.h"

#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0601        /* Windows 7: SRWLOCK, GetTickCount64, BCryptGenRandom */
#  endif
#  include <windows.h>
#  include <bcrypt.h>
#  include <process.h>
#else
#  include <dirent.h>
#  include <errno.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <time.h>
#  include <unistd.h>
#  if defined(__linux__) && defined(__GLIBC__) && \
      ((__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
#    include <sys/random.h>
#    define ENGRAM_HAVE_GETRANDOM 1
#  endif
#endif

/* ==================================================================================================
 * MUTEX
 * ============================================================================================== */
#ifdef _WIN32
ENGRAM_STATIC_ASSERT(sizeof(engram_mutex) == sizeof(SRWLOCK), srwlock_layout_matches);

void engram_mutex_lock(engram_mutex *m)   { AcquireSRWLockExclusive((PSRWLOCK)(void *)m); }
void engram_mutex_unlock(engram_mutex *m) { ReleaseSRWLockExclusive((PSRWLOCK)(void *)m); }
#else
void engram_mutex_lock(engram_mutex *m)   { (void)pthread_mutex_lock(&m->m); }
void engram_mutex_unlock(engram_mutex *m) { (void)pthread_mutex_unlock(&m->m); }
#endif

/* ==================================================================================================
 * IO FAULT INJECTION
 * ==================================================================================================
 * One counter, one lock. Every IO primitive below calls engram_io_tick() FIRST and fails cleanly if
 * it says so -- before touching anything -- so an injected failure is indistinguishable, to the
 * caller, from the operating system refusing. That is the point: the caller's error handling is what
 * is under test, and it must not be able to tell the difference. */
static engram_mutex g_io_lock = ENGRAM_MUTEX_INIT;
static uint64_t     g_io_ops = 0;
static uint64_t     g_io_fail_at = 0;
static uint64_t     g_io_injected = 0;

void engram_io_fail_at(uint64_t nth)
{
    engram_mutex_lock(&g_io_lock);
    g_io_fail_at = nth ? g_io_ops + nth : 0;
    engram_mutex_unlock(&g_io_lock);
}

uint64_t engram_io_ops(void)
{
    uint64_t v;
    engram_mutex_lock(&g_io_lock); v = g_io_ops; engram_mutex_unlock(&g_io_lock);
    return v;
}

uint64_t engram_io_injected(void)
{
    uint64_t v;
    engram_mutex_lock(&g_io_lock); v = g_io_injected; engram_mutex_unlock(&g_io_lock);
    return v;
}

void engram_io_reset(void)
{
    engram_mutex_lock(&g_io_lock);
    g_io_ops = 0; g_io_fail_at = 0; g_io_injected = 0;
    engram_mutex_unlock(&g_io_lock);
}

/* Returns 1 if THIS operation must fail. */
static int engram_io_tick(void)
{
    int fail = 0;
    engram_mutex_lock(&g_io_lock);
    g_io_ops++;
    if (g_io_fail_at && g_io_ops == g_io_fail_at) {
        fail = 1;
        g_io_injected++;
        g_io_fail_at = 0;           /* one failure, then the world behaves again */
    }
    engram_mutex_unlock(&g_io_lock);
    return fail;
}

/* ==================================================================================================
 * CLOCKS
 * ============================================================================================== */
#ifdef _WIN32
uint64_t engram_now_ns(void)
{
    /* The frequency is queried on EVERY call, deliberately. It used to be cached in a static written
     * without a lock, excused as "a benign race that writes the same value twice". There is no benign
     * data race in C: an unsynchronised write concurrent with a read is undefined behaviour, and an
     * optimiser is entitled to assume it cannot happen. The frequency is fixed at boot and the query
     * is cheap, so the shared state is simply removed rather than defended. */
    LARGE_INTEGER freq, c;
    QueryPerformanceFrequency(&freq);
    if (freq.QuadPart <= 0) return 0;
    QueryPerformanceCounter(&c);
    /* Split to avoid overflowing c * 1e9 on machines with a high counter frequency: the naive
     * product overflows 64 bits after ~30 minutes at 10 MHz. */
    return (uint64_t)(c.QuadPart / freq.QuadPart) * 1000000000ull +
           (uint64_t)(c.QuadPart % freq.QuadPart) * 1000000000ull / (uint64_t)freq.QuadPart;
}

int64_t engram_wall_ms(void)
{
    FILETIME ft;
    ULARGE_INTEGER u;
    GetSystemTimeAsFileTime(&ft);
    u.LowPart = ft.dwLowDateTime; u.HighPart = ft.dwHighDateTime;
    /* FILETIME counts 100 ns ticks since 1601-01-01; the Unix epoch is 11,644,473,600 s later. */
    return (int64_t)(u.QuadPart / 10000ull) - 11644473600000ll;
}
#else
uint64_t engram_now_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int64_t engram_wall_ms(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) != 0) return 0;
    return (int64_t)ts.tv_sec * 1000ll + (int64_t)(ts.tv_nsec / 1000000l);
}
#endif

/* Civil date from days since 1970-01-01, proleptic Gregorian. Howard Hinnant's algorithm: pure
 * integer arithmetic, correct for every date either side of the epoch, and identical on every
 * platform -- which gmtime_r and gmtime_s are not guaranteed to be at the edges. */
static void engram_civil_from_days(int64_t z, int64_t *y, unsigned *m, unsigned *d)
{
    int64_t era, yoe, yr, doy, mp;
    z += 719468;
    era = (z >= 0 ? z : z - 146096) / 146097;
    yoe = z - era * 146097;
    yr  = (yoe - yoe / 1460 + yoe / 36524 - yoe / 146096) / 365;
    doy = yoe - (365 * yr + yr / 4 - yr / 100);
    mp  = (5 * doy + 2) / 153;
    *d  = (unsigned)(doy - (153 * mp + 2) / 5 + 1);
    *m  = (unsigned)(mp < 10 ? mp + 3 : mp - 9);
    *y  = yr + era * 400 + (*m <= 2 ? 1 : 0);
}

engram_rc engram_wall_format(int64_t ms, char *buf, size_t cap)
{
    int64_t days, rem, y;
    unsigned mo, d, hh, mi, ss, mss;
    int n;
    if (!buf || cap < 25u) return ENGRAM_E_ARG;
    days = ms / 86400000ll;
    rem  = ms % 86400000ll;
    if (rem < 0) { rem += 86400000ll; days--; }
    engram_civil_from_days(days, &y, &mo, &d);
    hh  = (unsigned)(rem / 3600000ll);  rem %= 3600000ll;
    mi  = (unsigned)(rem / 60000ll);    rem %= 60000ll;
    ss  = (unsigned)(rem / 1000ll);
    mss = (unsigned)(rem % 1000ll);
    if (y < 0 || y > 9999) return ENGRAM_E_ARG;      /* the format has four year digits, no more */
    n = snprintf(buf, cap, "%04d-%02u-%02uT%02u:%02u:%02u.%03uZ", (int)y, mo, d, hh, mi, ss, mss);
    return (n > 0 && (size_t)n < cap) ? ENGRAM_OK : ENGRAM_E_ARG;
}

/* ==================================================================================================
 * PATHS (Win32): UTF-8 -> UTF-16, refusing invalid input
 * ============================================================================================== */
#ifdef _WIN32
static wchar_t *engram_widen(const char *utf8)
{
    int n;
    wchar_t *w;
    if (!utf8) return NULL;
    n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, NULL, 0);
    if (n <= 0) return NULL;
    w = (wchar_t *)engram_array((size_t)n, sizeof(wchar_t));
    if (!w) return NULL;
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, w, n) != n) {
        engram_free(w);
        return NULL;
    }
    return w;
}
#endif

/* The temporary name used by the atomic write: same directory as the target (rename is only atomic
 * within one filesystem), unique per process and per call so two writers never collide. */
static engram_rc engram_tmp_name(const char *path, char **out)
{
    static engram_mutex lock = ENGRAM_MUTEX_INIT;
    static uint64_t seq = 0;
    uint64_t s, pid;
    size_t n;
    char *t;
    engram_mutex_lock(&lock); s = ++seq; engram_mutex_unlock(&lock);
#ifdef _WIN32
    pid = (uint64_t)GetCurrentProcessId();
#else
    pid = (uint64_t)getpid();
#endif
    n = strlen(path) + 64u;
    t = (char *)engram_malloc(n);
    if (!t) return ENGRAM_E_MEM;
    snprintf(t, n, "%s.tmp.%llu.%llu", path, (unsigned long long)pid, (unsigned long long)s);
    *out = t;
    return ENGRAM_OK;
}

/* ==================================================================================================
 * FILES -- Win32
 * ============================================================================================== */
#ifdef _WIN32

engram_rc engram_file_read(const char *path, uint8_t **out, size_t *len)
{
    wchar_t *w;
    HANDLE h;
    LARGE_INTEGER sz;
    uint8_t *buf;
    size_t got = 0;
    if (out) *out = NULL;
    if (len) *len = 0;
    if (!path || !out || !len) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    w = engram_widen(path);
    if (!w) return ENGRAM_E_UTF8;
    h = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    engram_free(w);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        return (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
    }
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart < 0) { CloseHandle(h); return ENGRAM_E_IO; }
    if ((uint64_t)sz.QuadPart > (uint64_t)ENGRAM_FILE_MAX) { CloseHandle(h); return ENGRAM_E_OVERFLOW; }
    buf = (uint8_t *)engram_malloc((size_t)sz.QuadPart + 1u);
    if (!buf) { CloseHandle(h); return ENGRAM_E_MEM; }
    while (got < (size_t)sz.QuadPart) {
        DWORD want = (DWORD)(((size_t)sz.QuadPart - got) > 0x40000000u ? 0x40000000u
                                                                        : ((size_t)sz.QuadPart - got));
        DWORD n = 0;
        if (engram_io_tick() || !ReadFile(h, buf + got, want, &n, NULL)) {
            CloseHandle(h); engram_free(buf); return ENGRAM_E_IO;
        }
        if (n == 0) break;                         /* file shrank underneath us */
        got += n;
    }
    CloseHandle(h);
    if (got != (size_t)sz.QuadPart) { engram_free(buf); return ENGRAM_E_IO; }
    buf[got] = 0;
    *out = buf;
    *len = got;
    return ENGRAM_OK;
}

engram_rc engram_file_write_atomic(const char *path, const void *data, size_t len)
{
    char *tmp = NULL;
    wchar_t *wt = NULL, *wp = NULL;
    HANDLE h;
    const uint8_t *p = (const uint8_t *)data;
    size_t off = 0;
    engram_rc rc;
    if (!path || (!data && len)) return ENGRAM_E_ARG;
    rc = engram_tmp_name(path, &tmp);
    if (rc != ENGRAM_OK) return rc;
    wt = engram_widen(tmp);
    wp = engram_widen(path);
    if (!wt || !wp) { rc = ENGRAM_E_UTF8; goto done; }

    /* 1. write the complete contents to a fresh temporary. CREATE_NEW: never reuse a stale one. */
    if (engram_io_tick()) { rc = ENGRAM_E_IO; goto done; }
    h = CreateFileW(wt, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) { rc = ENGRAM_E_IO; goto done; }
    while (off < len) {
        DWORD want = (DWORD)((len - off) > 0x40000000u ? 0x40000000u : (len - off));
        DWORD n = 0;
        if (engram_io_tick() || !WriteFile(h, p + off, want, &n, NULL) || n == 0) {
            CloseHandle(h); DeleteFileW(wt); rc = ENGRAM_E_IO; goto done;
        }
        off += n;
    }
    /* 2. flush to the device. */
    if (engram_io_tick() || !FlushFileBuffers(h)) {
        CloseHandle(h); DeleteFileW(wt); rc = ENGRAM_E_IO; goto done;
    }
    CloseHandle(h);
    /* 3 + 4. rename over the target; WRITE_THROUGH does not return until the rename is durable. */
    if (engram_io_tick() ||
        !MoveFileExW(wt, wp, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        DeleteFileW(wt); rc = ENGRAM_E_IO; goto done;
    }
    rc = ENGRAM_OK;
done:
    engram_free(wt); engram_free(wp); engram_free(tmp);
    return rc;
}

int engram_file_exists(const char *path)
{
    wchar_t *w;
    WIN32_FILE_ATTRIBUTE_DATA a;
    BOOL ok;
    if (!path) return 0;
    w = engram_widen(path);
    if (!w) return 0;
    ok = GetFileAttributesExW(w, GetFileExInfoStandard, &a);
    engram_free(w);
    return ok && !(a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY);
}

engram_rc engram_file_size(const char *path, uint64_t *size)
{
    wchar_t *w;
    WIN32_FILE_ATTRIBUTE_DATA a;
    BOOL ok;
    if (size) *size = 0;
    if (!path || !size) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    w = engram_widen(path);
    if (!w) return ENGRAM_E_UTF8;
    ok = GetFileAttributesExW(w, GetFileExInfoStandard, &a);
    engram_free(w);
    if (!ok || (a.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) return ENGRAM_E_NOTFOUND;
    *size = ((uint64_t)a.nFileSizeHigh << 32) | (uint64_t)a.nFileSizeLow;
    return ENGRAM_OK;
}

engram_rc engram_file_remove(const char *path)
{
    wchar_t *w;
    BOOL ok;
    DWORD e;
    if (!path) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    w = engram_widen(path);
    if (!w) return ENGRAM_E_UTF8;
    ok = DeleteFileW(w);
    e = ok ? 0 : GetLastError();
    engram_free(w);
    if (ok) return ENGRAM_OK;
    return (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
}

engram_rc engram_dir_make(const char *path)
{
    wchar_t *w;
    BOOL ok;
    DWORD e;
    if (!path) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    w = engram_widen(path);
    if (!w) return ENGRAM_E_UTF8;
    ok = CreateDirectoryW(w, NULL);
    e = ok ? 0 : GetLastError();
    engram_free(w);
    return (ok || e == ERROR_ALREADY_EXISTS) ? ENGRAM_OK : ENGRAM_E_IO;
}

#else
/* ==================================================================================================
 * FILES -- POSIX
 * ============================================================================================== */

engram_rc engram_file_read(const char *path, uint8_t **out, size_t *len)
{
    int fd;
    struct stat st;
    uint8_t *buf;
    size_t got = 0;
    if (out) *out = NULL;
    if (len) *len = 0;
    if (!path || !out || !len) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    do { fd = open(path, O_RDONLY); } while (fd < 0 && errno == EINTR);
    if (fd < 0) return errno == ENOENT ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) { close(fd); return ENGRAM_E_IO; }
    if ((uint64_t)st.st_size > (uint64_t)ENGRAM_FILE_MAX) { close(fd); return ENGRAM_E_OVERFLOW; }
    buf = (uint8_t *)engram_malloc((size_t)st.st_size + 1u);
    if (!buf) { close(fd); return ENGRAM_E_MEM; }
    while (got < (size_t)st.st_size) {
        ssize_t n;
        if (engram_io_tick()) { close(fd); engram_free(buf); return ENGRAM_E_IO; }
        do { n = read(fd, buf + got, (size_t)st.st_size - got); } while (n < 0 && errno == EINTR);
        if (n < 0) { close(fd); engram_free(buf); return ENGRAM_E_IO; }
        if (n == 0) break;                          /* file shrank underneath us */
        got += (size_t)n;
    }
    close(fd);
    if (got != (size_t)st.st_size) { engram_free(buf); return ENGRAM_E_IO; }
    buf[got] = 0;
    *out = buf;
    *len = got;
    return ENGRAM_OK;
}

/* The directory containing `path`, for step 4. "." when there is no separator. */
static char *engram_dirname_dup(const char *path)
{
    const char *slash = strrchr(path, '/');
    size_t n;
    char *d;
    if (!slash) {
        d = (char *)engram_malloc(2u);
        if (d) { d[0] = '.'; d[1] = 0; }
        return d;
    }
    n = (size_t)(slash - path);
    if (n == 0) n = 1;                              /* "/file" -> "/" */
    d = (char *)engram_malloc(n + 1u);
    if (!d) return NULL;
    memcpy(d, path, n);
    d[n] = 0;
    return d;
}

static int engram_write_all(int fd, const uint8_t *p, size_t len)
{
    size_t off = 0;
    while (off < len) {
        ssize_t n;
        if (engram_io_tick()) return -1;
        do { n = write(fd, p + off, len - off); } while (n < 0 && errno == EINTR);
        if (n <= 0) return -1;
        off += (size_t)n;
    }
    return 0;
}

engram_rc engram_file_write_atomic(const char *path, const void *data, size_t len)
{
    char *tmp = NULL, *dir = NULL;
    int fd, dfd;
    engram_rc rc;
    if (!path || (!data && len)) return ENGRAM_E_ARG;
    rc = engram_tmp_name(path, &tmp);
    if (rc != ENGRAM_OK) return rc;

    /* 1. write the complete contents to a fresh temporary. O_EXCL: never reuse a stale one. */
    if (engram_io_tick()) { rc = ENGRAM_E_IO; goto done; }
    do { fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0600); } while (fd < 0 && errno == EINTR);
    if (fd < 0) { rc = ENGRAM_E_IO; goto done; }
    if (engram_write_all(fd, (const uint8_t *)data, len) != 0) {
        close(fd); unlink(tmp); rc = ENGRAM_E_IO; goto done;
    }
    /* 2. flush to the device. */
    if (engram_io_tick() || fsync(fd) != 0) { close(fd); unlink(tmp); rc = ENGRAM_E_IO; goto done; }
    if (close(fd) != 0) { unlink(tmp); rc = ENGRAM_E_IO; goto done; }
    /* 3. rename over the target. */
    if (engram_io_tick() || rename(tmp, path) != 0) { unlink(tmp); rc = ENGRAM_E_IO; goto done; }
    /* 4. flush the directory entry, so the rename itself survives a power cut. */
    dir = engram_dirname_dup(path);
    if (!dir) { rc = ENGRAM_E_MEM; goto done; }       /* the file IS written; durability unknown */
    if (engram_io_tick()) { rc = ENGRAM_E_IO; goto done; }
    do { dfd = open(dir, O_RDONLY); } while (dfd < 0 && errno == EINTR);
    if (dfd >= 0) {
        int r = fsync(dfd);
        close(dfd);
        /* EINVAL: this filesystem does not support syncing a directory (some network and FUSE
         * mounts). The data is flushed and the rename done; there is nothing more that CAN be done,
         * and refusing would make the store unwritable on that filesystem forever. */
        if (r != 0 && errno != EINVAL) { rc = ENGRAM_E_IO; goto done; }
    }
    rc = ENGRAM_OK;
done:
    engram_free(dir);
    engram_free(tmp);
    return rc;
}

int engram_file_exists(const char *path)
{
    struct stat st;
    if (!path) return 0;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

engram_rc engram_file_size(const char *path, uint64_t *size)
{
    struct stat st;
    if (size) *size = 0;
    if (!path || !size) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    if (stat(path, &st) != 0) return errno == ENOENT ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
    if (!S_ISREG(st.st_mode)) return ENGRAM_E_NOTFOUND;
    *size = (uint64_t)st.st_size;
    return ENGRAM_OK;
}

engram_rc engram_file_remove(const char *path)
{
    if (!path) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    if (unlink(path) == 0) return ENGRAM_OK;
    return errno == ENOENT ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
}

engram_rc engram_dir_make(const char *path)
{
    if (!path) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    if (mkdir(path, 0700) == 0 || errno == EEXIST) return ENGRAM_OK;
    return ENGRAM_E_IO;
}
#endif

/* ==================================================================================================
 * PATHS AND DIRECTORIES
 * ============================================================================================== */
char *engram_path_join(const char *dir, const char *name)
{
    size_t a, b;
    int sep;
    char *p;
    if (!dir || !name) return NULL;
    a = strlen(dir);
    b = strlen(name);
    if (b > SIZE_MAX - a - 2u) return NULL;
    sep = a > 0u && dir[a - 1u] != '/' && dir[a - 1u] != '\\';
    p = (char *)engram_malloc(a + (size_t)sep + b + 1u);
    if (!p) return NULL;
    memcpy(p, dir, a);
    if (sep) p[a++] = '/';
    memcpy(p + a, name, b + 1u);
    return p;
}

#ifdef _WIN32
static char *engram_narrow(const wchar_t *w)
{
    int n;
    char *s;
    n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, NULL, 0, NULL, NULL);
    if (n <= 0) return NULL;
    s = (char *)engram_malloc((size_t)n);
    if (!s) return NULL;
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, s, n, NULL, NULL) != n) {
        engram_free(s);
        return NULL;
    }
    return s;
}

engram_rc engram_dir_list(const char *dir, engram_dir_cb cb, void *user)
{
    char *pat;
    wchar_t *wpat;
    HANDLE h;
    WIN32_FIND_DATAW fd;
    DWORD e;
    int stopped = 0;
    if (!dir || !cb) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    pat = engram_path_join(dir, "*");
    if (!pat) return ENGRAM_E_MEM;
    wpat = engram_widen(pat);
    engram_free(pat);
    if (!wpat) return ENGRAM_E_UTF8;
    h = FindFirstFileW(wpat, &fd);
    engram_free(wpat);
    if (h == INVALID_HANDLE_VALUE) {
        e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND) return ENGRAM_OK;          /* an empty directory */
        return e == ERROR_PATH_NOT_FOUND ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
    }
    do {
        char *name;
        if (fd.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_DEVICE)) continue;
        name = engram_narrow(fd.cFileName);
        if (!name) {
            /* NTFS permits unpaired surrogates in names, which have no UTF-8 form. Such a file is
             * skipped VISIBLY -- logged, counted by the log -- rather than silently. */
            engram_log(ENGRAM_LOG_WARN, "plat", "skipped a file in %s whose name is not valid UTF-16", dir);
            continue;
        }
        if (cb(name, user)) { engram_free(name); stopped = 1; break; }
        engram_free(name);
    } while (FindNextFileW(h, &fd));
    e = stopped ? ERROR_NO_MORE_FILES : GetLastError();
    FindClose(h);
    return e == ERROR_NO_MORE_FILES ? ENGRAM_OK : ENGRAM_E_IO;
}
#else
engram_rc engram_dir_list(const char *dir, engram_dir_cb cb, void *user)
{
    DIR *d;
    struct dirent *e;
    if (!dir || !cb) return ENGRAM_E_ARG;
    if (engram_io_tick()) return ENGRAM_E_IO;
    d = opendir(dir);
    if (!d) return errno == ENOENT ? ENGRAM_E_NOTFOUND : ENGRAM_E_IO;
    for (;;) {
        struct stat st;
        char *path;
        int regular;
        errno = 0;
        e = readdir(d);
        if (!e) {
            if (errno) { closedir(d); return ENGRAM_E_IO; }
            break;
        }
        if (e->d_name[0] == '.' &&
            (e->d_name[1] == 0 || (e->d_name[1] == '.' && e->d_name[2] == 0))) continue;
        /* d_type is DT_UNKNOWN on some filesystems, so the type is asked of stat, not of readdir. */
        path = engram_path_join(dir, e->d_name);
        if (!path) { closedir(d); return ENGRAM_E_MEM; }
        regular = stat(path, &st) == 0 && S_ISREG(st.st_mode);
        engram_free(path);
        if (!regular) continue;
        if (cb(e->d_name, user)) break;
    }
    closedir(d);
    return ENGRAM_OK;
}
#endif

int engram_tmp_name_is_orphan(const char *name)
{
    const char *p, *t, *last = NULL;
    size_t nd;
    if (!name) return 0;
    for (p = name; (t = strstr(p, ".tmp.")) != NULL; p = t + 1) last = t;
    if (!last || last == name) return 0;                   /* no marker, or an empty base */
    p = last + 5;
    for (nd = 0; *p >= '0' && *p <= '9'; p++) nd++;
    if (!nd || *p != '.') return 0;
    p++;
    for (nd = 0; *p >= '0' && *p <= '9'; p++) nd++;
    return nd > 0u && *p == 0;
}

typedef struct {
    char   **names;
    size_t   n, cap;
    engram_rc err;
} engram_sweep_list;

static int engram_sweep_collect(const char *name, void *user)
{
    engram_sweep_list *L = (engram_sweep_list *)user;
    char *copy;
    engram_rc rc;
    if (!engram_tmp_name_is_orphan(name)) return 0;
    rc = engram_grow((void **)&L->names, &L->cap, L->n + 1u, sizeof *L->names);
    if (rc != ENGRAM_OK) { L->err = rc; return 1; }
    copy = engram_strdup(name);
    if (!copy) { L->err = ENGRAM_E_MEM; return 1; }
    L->names[L->n++] = copy;
    return 0;
}

engram_rc engram_tmp_sweep(const char *dir, unsigned *removed)
{
    engram_sweep_list L;
    engram_rc rc, first = ENGRAM_OK;
    size_t i;
    if (removed) *removed = 0;
    if (!dir) return ENGRAM_E_ARG;
    memset(&L, 0, sizeof L);
    L.err = ENGRAM_OK;
    /* Collect first, remove after: deleting entries while a directory iterator is open has
     * platform-specific semantics, and none of them is worth depending on. */
    rc = engram_dir_list(dir, engram_sweep_collect, &L);
    if (rc == ENGRAM_OK) rc = L.err;
    for (i = 0; i < L.n; i++) {
        if (rc == ENGRAM_OK) {
            char *path = engram_path_join(dir, L.names[i]);
            engram_rc r = path ? engram_file_remove(path) : ENGRAM_E_MEM;
            engram_free(path);
            if (r == ENGRAM_OK) { if (removed) (*removed)++; }
            else if (r != ENGRAM_E_NOTFOUND && first == ENGRAM_OK) first = r;
        }
        engram_free(L.names[i]);
    }
    engram_free(L.names);
    return rc != ENGRAM_OK ? rc : first;
}

/* ==================================================================================================
 * IDLE
 * ============================================================================================== */
engram_rc engram_idle_query(engram_idle *out)
{
    if (!out) return ENGRAM_E_ARG;
    memset(out, 0, sizeof *out);
    out->on_mains = -1;
    out->battery_pct = -1;
#ifdef _WIN32
    {
        LASTINPUTINFO lii;
        SYSTEM_POWER_STATUS ps;
        lii.cbSize = sizeof lii;
        if (GetLastInputInfo(&lii)) {
            /* Both are 32-bit tick counts that wrap every 49.7 days. Unsigned subtraction gives the
             * right interval across the wrap; converting either to 64 bits first would not. */
            DWORD now = GetTickCount();
            out->known = 1;
            out->idle_ms = (uint64_t)(DWORD)(now - lii.dwTime);
        }
        if (GetSystemPowerStatus(&ps)) {
            out->on_mains    = ps.ACLineStatus == 1 ? 1 : (ps.ACLineStatus == 0 ? 0 : -1);
            out->battery_pct = ps.BatteryLifePercent <= 100 ? (int)ps.BatteryLifePercent : -1;
        }
    }
#endif
    return ENGRAM_OK;
}

/* ==================================================================================================
 * ENTROPY
 * ============================================================================================== */
engram_rc engram_os_random(void *out, size_t n)
{
    uint8_t *p = (uint8_t *)out;
    if (!out && n) return ENGRAM_E_ARG;
#ifdef _WIN32
    while (n) {
        ULONG chunk = (ULONG)(n > 0x10000000u ? 0x10000000u : n);
        if (!BCRYPT_SUCCESS(BCryptGenRandom(NULL, p, chunk, BCRYPT_USE_SYSTEM_PREFERRED_RNG)))
            return ENGRAM_E_IO;
        p += chunk; n -= chunk;
    }
    return ENGRAM_OK;
#else
#  ifdef ENGRAM_HAVE_GETRANDOM
    while (n) {
        ssize_t r = getrandom(p, n, 0);
        if (r < 0) { if (errno == EINTR) continue; break; }
        p += (size_t)r; n -= (size_t)r;
    }
    if (n == 0) return ENGRAM_OK;
#  endif
    {
        int fd;
        do { fd = open("/dev/urandom", O_RDONLY); } while (fd < 0 && errno == EINTR);
        if (fd < 0) return ENGRAM_E_IO;
        while (n) {
            ssize_t r;
            do { r = read(fd, p, n); } while (r < 0 && errno == EINTR);
            if (r <= 0) { close(fd); return ENGRAM_E_IO; }
            p += (size_t)r; n -= (size_t)r;
        }
        close(fd);
        return ENGRAM_OK;
    }
#endif
}

/* ==================================================================================================
 * THREADS
 * ============================================================================================== */
struct engram_thread {
    engram_thread_fn fn;
    void            *arg;
#ifdef _WIN32
    HANDLE           h;
#else
    pthread_t        t;
#endif
};

#ifdef _WIN32
static unsigned __stdcall engram_thread_tramp(void *p)
{
    engram_thread *t = (engram_thread *)p;
    t->fn(t->arg);
    return 0u;
}
#else
static void *engram_thread_tramp(void *p)
{
    engram_thread *t = (engram_thread *)p;
    t->fn(t->arg);
    return NULL;
}
#endif

engram_rc engram_thread_start(engram_thread **out, engram_thread_fn fn, void *arg)
{
    engram_thread *t;
    if (out) *out = NULL;
    if (!out || !fn) return ENGRAM_E_ARG;
    t = (engram_thread *)engram_calloc(1u, sizeof *t);
    if (!t) return ENGRAM_E_MEM;
    t->fn = fn;
    t->arg = arg;
#ifdef _WIN32
    {
        uintptr_t h = _beginthreadex(NULL, 0u, engram_thread_tramp, t, 0u, NULL);
        if (h == 0u) { engram_free(t); return ENGRAM_E_IO; }
        t->h = (HANDLE)h;
    }
#else
    if (pthread_create(&t->t, NULL, engram_thread_tramp, t) != 0) { engram_free(t); return ENGRAM_E_IO; }
#endif
    *out = t;
    return ENGRAM_OK;
}

engram_rc engram_thread_join(engram_thread *t)
{
    engram_rc rc = ENGRAM_OK;
    if (!t) return ENGRAM_E_ARG;
#ifdef _WIN32
    if (WaitForSingleObject(t->h, INFINITE) != WAIT_OBJECT_0) rc = ENGRAM_E_IO;
    CloseHandle(t->h);
#else
    if (pthread_join(t->t, NULL) != 0) rc = ENGRAM_E_IO;
#endif
    engram_free(t);
    return rc;
}

/* ==================================================================================================
 * HOST
 * ============================================================================================== */
unsigned engram_cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors ? (unsigned)si.dwNumberOfProcessors : 1u;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (unsigned)n : 1u;
#endif
}
