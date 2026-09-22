/* ==================================================================================================
 * engram_plat.h -- the operating system, and only the parts ENGRAM actually uses.
 * ==================================================================================================
 *
 * Two implementations, POSIX and Win32, behind one interface. Not a portability library: a clock, a
 * lock, whole-file reads, atomic whole-file writes, and the idle signal the sleep scheduler needs.
 * Anything wider would be surface with no caller.
 *
 * ==================================================================================================
 * WHY WRITES ARE WHOLE-FILE AND ATOMIC, AND WHAT "ATOMIC" HAS TO MEAN
 * ==================================================================================================
 * ENGRAM's files are its memory. A torn write -- half an old store and half a new one -- is not a
 * corrupt file to be repaired, it is a user's memory quietly replaced by garbage that may still
 * parse. So there is exactly one way to write a file, and it cannot tear:
 *
 *      1  write the complete new contents to a temporary file in the SAME directory
 *      2  flush it to the device        (fsync / FlushFileBuffers)
 *      3  rename it over the target     (rename / MoveFileExW REPLACE_EXISTING|WRITE_THROUGH)
 *      4  flush the directory entry     (POSIX: fsync on the directory; Win32: WRITE_THROUGH did it)
 *
 * A crash before step 3 leaves the old file untouched and a stray temporary. A crash after step 3
 * leaves the new file. There is no instant at which the target holds anything else.
 *
 * Step 1 is in the SAME directory because rename is only atomic within one filesystem, and a
 * temporary in /tmp would silently turn the rename into copy-then-delete on some systems.
 *
 * Step 4 is the one most implementations omit, and without it a power cut can lose the RENAME even
 * though the data was flushed -- the new contents are durable and nothing points at them.
 *
 * ==================================================================================================
 * PATHS ARE UTF-8, ON EVERY PLATFORM
 * ==================================================================================================
 * On Windows every path is converted to UTF-16 and passed to the W functions. The A functions use
 * the system code page, which silently corrupts any path outside it -- a user whose Documents folder
 * has a non-Latin name would get "file not found" for a file that plainly exists. Conversion refuses
 * invalid UTF-8 rather than substituting replacement characters, because a substituted path names a
 * DIFFERENT file.
 * ============================================================================================== */
#ifndef ENGRAM_PLAT_H
#define ENGRAM_PLAT_H

#include "engram.h"

#ifndef _WIN32
#  include <pthread.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- A MUTEX THAT NEEDS NO INITIALISATION CALL -----------------------------------------------
 * Statically initialisable on both platforms, so a global lock needs no init ordering and there is
 * no window in which it exists uninitialised.
 *
 * Win32: SRWLOCK is `struct { PVOID Ptr; }` with SRWLOCK_INIT == {0}. This struct has the same
 * layout and the .c file static-asserts that it does, so the cast there is between identical
 * layouts and not a guess. Exclusive mode only -- nothing here needs a reader/writer split, and
 * an unused capability is surface. */
#ifdef _WIN32
typedef struct { void *opaque; } engram_mutex;
#  define ENGRAM_MUTEX_INIT { 0 }
#else
typedef struct { pthread_mutex_t m; } engram_mutex;
#  define ENGRAM_MUTEX_INIT { PTHREAD_MUTEX_INITIALIZER }
#endif

void engram_mutex_lock(engram_mutex *m);
void engram_mutex_unlock(engram_mutex *m);

/* ---- CLOCKS ----------------------------------------------------------------------------------
 * MONOTONIC for measuring durations: never goes backwards, unaffected by the user changing the time
 * or by NTP. WALL for timestamps a human reads. Mixing them is a classic defect -- a duration
 * measured on the wall clock goes negative across a daylight-saving change. */
uint64_t engram_now_ns(void);     /* monotonic nanoseconds since an arbitrary fixed origin */
int64_t  engram_wall_ms(void);    /* milliseconds since the Unix epoch, UTC                */

/* Format a wall-clock millisecond time as "YYYY-MM-DDTHH:MM:SS.mmmZ" into buf (needs >= 25 bytes).
 * Returns ENGRAM_E_ARG if the buffer is too small rather than writing a truncated timestamp that
 * would sort wrongly in a log. */
engram_rc engram_wall_format(int64_t ms, char *buf, size_t cap);

/* ---- FILES -----------------------------------------------------------------------------------
 * Every call counts toward the IO fault injector (engram_io_fail_at), so every failure path in every
 * caller can be exercised deliberately rather than trusted. */

/* The largest file engram_file_read will load. A store is a bounded thing; a multi-gigabyte file
 * where a store should be is an error to report, not an allocation to attempt. */
#ifndef ENGRAM_FILE_MAX
#define ENGRAM_FILE_MAX ((size_t)1u << 31)     /* 2 GiB */
#endif

/* Read an entire file. On success *out is an engram_alloc'd buffer the caller frees with
 * engram_free, and *len its length. A trailing NUL is appended (not counted in *len) so text can be
 * read as a string without a second copy. On any failure *out is NULL and *len is 0. */
engram_rc engram_file_read(const char *path, uint8_t **out, size_t *len);

/* Write an entire file ATOMICALLY (the four steps above). The target either keeps its old contents
 * or holds exactly `data`, never anything in between. */
engram_rc engram_file_write_atomic(const char *path, const void *data, size_t len);

/* 1 if the path exists and is a regular file, 0 if not. Never fails: "cannot tell" is "no". */
int engram_file_exists(const char *path);

/* Size of a regular file. ENGRAM_E_NOTFOUND if absent. */
engram_rc engram_file_size(const char *path, uint64_t *size);

/* Remove a file. Removing a file that does not exist is ENGRAM_E_NOTFOUND, not success -- a caller
 * that believed it was deleting something should learn that it was not there. */
engram_rc engram_file_remove(const char *path);

/* dir + separator + name, as a new engram_alloc'd string. NULL on failure. A separator is added only
 * when `dir` does not already end in one. '/' is used everywhere; Win32 accepts it. */
char *engram_path_join(const char *dir, const char *name);

/* Create a directory if absent. Existing is success. Not recursive. */
engram_rc engram_dir_make(const char *path);

/* Call `cb` once per REGULAR FILE directly inside `dir` -- not recursive; directories, links to
 * directories and special files are skipped -- passing the entry's NAME (not its path), UTF-8. The
 * callback returns 0 to continue, nonzero to stop early (which is still ENGRAM_OK). The order is
 * whatever the filesystem returns and is NOT stable across machines; a caller that needs determinism
 * must sort, because a measurement over "the files in a folder" in readdir order is a measurement
 * over a different corpus on every copy of that folder. */
typedef int (*engram_dir_cb)(const char *name, void *user);
engram_rc engram_dir_list(const char *dir, engram_dir_cb cb, void *user);

/* ---- ORPHANED TEMPORARIES --------------------------------------------------------------------
 * The atomic writer cleans up its temporary on every FAILURE it can see. It cannot clean up after a
 * process that was KILLED between creating the temporary and renaming it -- power cut, task manager,
 * crash in another thread. Those orphans are harmless to correctness (the target was never touched)
 * but they accumulate, and a folder slowly filling with dead fragments of a user's memory is not
 * acceptable in software that runs for years.
 *
 * So a store sweeps its own directory when it opens. The match is STRICT -- "<base>.tmp.<digits>.
 * <digits>" with a non-empty base -- so no file the user created can be mistaken for an orphan.
 * Sweeping while another process is mid-write to the same directory deletes that process's temporary;
 * its rename then fails and its target is left intact, so atomicity holds and the cost is one clean,
 * retryable refusal. Call it only on a directory this process owns. */
int       engram_tmp_name_is_orphan(const char *name);
engram_rc engram_tmp_sweep(const char *dir, unsigned *removed);

/* ---- IO FAULT INJECTION ----------------------------------------------------------------------
 * Fail the Nth IO operation from now (1-based), then continue normally. 0 disables. Every open,
 * read, write, flush and rename is one operation. Used by the fault sweep to prove that every IO
 * failure anywhere in the tree is a clean refusal and never a corrupt store. */
void     engram_io_fail_at(uint64_t nth);
uint64_t engram_io_ops(void);            /* operations counted since the last reset   */
uint64_t engram_io_injected(void);       /* failures deliberately injected            */
void     engram_io_reset(void);

/* ---- THE IDLE SIGNAL -------------------------------------------------------------------------
 * What the sleep scheduler reads. Win32: GetLastInputInfo and GetSystemPowerStatus. POSIX has no
 * portable equivalent, so it reports "unknown" rather than inventing a number -- the scheduler then
 * falls back to an explicit `sleep` command, which is honest, rather than to a guess that would start
 * consolidating while the user is typing. */
typedef struct {
    int      known;          /* 1 if the platform can report idle time at all                     */
    uint64_t idle_ms;        /* milliseconds since the last user input                            */
    int      on_mains;       /* 1 mains, 0 battery, -1 unknown                                     */
    int      battery_pct;    /* 0..100, or -1 unknown                                              */
} engram_idle;

engram_rc engram_idle_query(engram_idle *out);

/* ---- ENTROPY ---------------------------------------------------------------------------------
 * Cryptographic random bytes from the operating system: BCryptGenRandom on Windows, getrandom or
 * /dev/urandom on POSIX. Failure is REPORTED -- a key generated from a failed entropy call and
 * silently zero-filled would be the most dangerous bug this program could have. */
engram_rc engram_os_random(void *out, size_t n);

/* ---- THREADS ---------------------------------------------------------------------------------
 * The minimum a deterministic parallel trainer needs (P2.2): start a function on a thread, wait for it.
 * No detach, no cancel, no thread-local storage -- each is a way for a worker's lifetime or state to
 * escape the fixed partition that makes a threaded run reproducible.
 *
 * Win32 uses _beginthreadex rather than CreateThread: the worker calls CRT functions, and
 * _beginthreadex is the entry point the CRT documents as safe for that. */
typedef struct engram_thread engram_thread;
typedef void (*engram_thread_fn)(void *arg);

engram_rc engram_thread_start(engram_thread **out, engram_thread_fn fn, void *arg);

/* Wait for the thread to finish and release it. After this returns, *t is invalid. */
engram_rc engram_thread_join(engram_thread *t);

/* ---- HOST INFORMATION ------------------------------------------------------------------------ */
unsigned engram_cpu_count(void);         /* logical CPUs, at least 1 */

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_PLAT_H */
