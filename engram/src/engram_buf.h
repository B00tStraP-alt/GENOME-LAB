/* ==================================================================================================
 * engram_buf.h -- bytes in and bytes out, little-endian, bounded, and impossible to misuse quietly.
 * ==================================================================================================
 *
 * EVERY FILE ENGRAM WRITES IS BUILT HERE AND EVERY FILE IT READS IS PARSED HERE. That makes this the
 * attack surface: a store on disk is untrusted input the moment anything other than this program could
 * have touched it, and a parser that trusts a length field is how a corrupt file becomes a crash, or
 * worse, a successful load of garbage.
 *
 * ==================================================================================================
 * THE WRITER: STICKY ERRORS, AND THE ONE EXIT
 * ==================================================================================================
 * A serialiser checking the return of every u32 it writes is a serialiser nobody can read, and the
 * checks are exactly what gets dropped under time pressure. So the writer is STICKY: the first failure
 * is recorded and every later write becomes a no-op. The report is deferred -- and rule R1 permits
 * that for one reason only: the ONLY way to take bytes out is engram_wbuf_finish(), which returns the
 * recorded rc. The failure cannot be ignored; it can only be postponed to the point where the data
 * would be used.
 *
 * ==================================================================================================
 * THE READER: BOUNDED, STICKY, AND IT MUST BE FINISHED
 * ==================================================================================================
 * Every read is checked against the remaining length BEFORE the bytes are touched. An overrun sets a
 * sticky ENGRAM_E_FORMAT and returns ZERO (never stale or partial data), and every later read also
 * returns zero, so a truncated file decodes to zeros that the caller's validation will reject rather
 * than to a plausible prefix.
 *
 * engram_rbuf_end() FAILS if any byte is left unconsumed. Trailing bytes mean the file is not the
 * layout the reader thinks it is -- a newer version, a concatenation, a truncation that happened to
 * land on a field boundary -- and a reader that stops early and reports success has parsed a
 * different file from the one on disk.
 *
 * ==================================================================================================
 * ENDIANNESS
 * ==================================================================================================
 * Every multi-byte value is composed and decomposed byte by byte, so the host's byte order never
 * matters and never appears. A store written on a little-endian laptop reads identically on a
 * big-endian machine. Floats travel as their IEEE-754 bit pattern through a memcpy -- never through a
 * pointer cast, which strict aliasing permits the optimiser to reorder around.
 * ============================================================================================== */
#ifndef ENGRAM_BUF_H
#define ENGRAM_BUF_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- WRITER ---------------------------------------------------------------------------------- */
typedef struct {
    uint8_t  *p;
    size_t    n;         /* bytes written       */
    size_t    cap;       /* bytes allocated     */
    engram_rc err;       /* first failure, sticky; ENGRAM_OK while healthy */
} engram_wbuf;

void engram_wbuf_init(engram_wbuf *b);
void engram_wbuf_free(engram_wbuf *b);        /* safe on an initialised or finished buffer */

void engram_wbuf_reserve(engram_wbuf *b, size_t extra);
void engram_wbuf_bytes(engram_wbuf *b, const void *src, size_t n);
void engram_wbuf_zeros(engram_wbuf *b, size_t n);
void engram_wbuf_u8 (engram_wbuf *b, uint8_t  v);
void engram_wbuf_u16(engram_wbuf *b, uint16_t v);
void engram_wbuf_u32(engram_wbuf *b, uint32_t v);
void engram_wbuf_u64(engram_wbuf *b, uint64_t v);
void engram_wbuf_i32(engram_wbuf *b, int32_t  v);
void engram_wbuf_i64(engram_wbuf *b, int64_t  v);
void engram_wbuf_f32(engram_wbuf *b, float    v);
void engram_wbuf_f64(engram_wbuf *b, double   v);
void engram_wbuf_f32s(engram_wbuf *b, const float *v, size_t n);

/* A length-prefixed string: u32 byte count, then the bytes, no terminator. A string longer than
 * UINT32_MAX sets ENGRAM_E_OVERFLOW rather than writing a wrapped length. */
void engram_wbuf_str(engram_wbuf *b, const char *s);
void engram_wbuf_blob(engram_wbuf *b, const void *p, size_t n);

/* Overwrite an EARLIER position -- for a length or checksum known only after what it describes has
 * been written. Out of range sets ENGRAM_E_INTERNAL: patching past the end is a bug, not bad input. */
void engram_wbuf_put_u32_at(engram_wbuf *b, size_t off, uint32_t v);
void engram_wbuf_put_u64_at(engram_wbuf *b, size_t off, uint64_t v);

size_t    engram_wbuf_len(const engram_wbuf *b);
engram_rc engram_wbuf_status(const engram_wbuf *b);

/* THE ONLY EXIT. On ENGRAM_OK, ownership of the bytes moves to *out (free with engram_free) and the
 * buffer is left empty. On failure *out is NULL, *len 0, the buffer is freed, and the recorded rc is
 * returned. A zero-length success returns a valid non-NULL pointer. */
engram_rc engram_wbuf_finish(engram_wbuf *b, uint8_t **out, size_t *len);

/* ---- READER ---------------------------------------------------------------------------------- */
typedef struct {
    const uint8_t *p;
    size_t         n;
    size_t         off;
    engram_rc      err;
} engram_rbuf;

void engram_rbuf_init(engram_rbuf *r, const void *p, size_t n);

/* Copy n bytes out. On overrun: dst is ZEROED, the error set, 0 returned. 1 on success. */
int engram_rbuf_bytes(engram_rbuf *r, void *dst, size_t n);

/* A pointer to the next n bytes without copying, or NULL on overrun (error set). Valid for the life of
 * the underlying buffer. */
const uint8_t *engram_rbuf_view(engram_rbuf *r, size_t n);

int engram_rbuf_skip(engram_rbuf *r, size_t n);

uint8_t  engram_rbuf_u8 (engram_rbuf *r);
uint16_t engram_rbuf_u16(engram_rbuf *r);
uint32_t engram_rbuf_u32(engram_rbuf *r);
uint64_t engram_rbuf_u64(engram_rbuf *r);
int32_t  engram_rbuf_i32(engram_rbuf *r);
int64_t  engram_rbuf_i64(engram_rbuf *r);
float    engram_rbuf_f32(engram_rbuf *r);
double   engram_rbuf_f64(engram_rbuf *r);

/* A FINITE float. NaN or infinity sets ENGRAM_E_FORMAT and returns 0 -- a non-finite weight in a
 * store is corruption, and a network fed one produces NaN everywhere downstream while every counter
 * keeps counting. */
float engram_rbuf_f32_finite(engram_rbuf *r);
int   engram_rbuf_f32s_finite(engram_rbuf *r, float *dst, size_t n);

/* A length-prefixed string, refused (NULL, ENGRAM_E_FORMAT) if its declared length exceeds maxlen or
 * the bytes remaining. Returns an engram_alloc'd NUL-terminated copy; NULL + ENGRAM_E_MEM if the copy
 * cannot be allocated. A string containing an embedded NUL is refused: it would read back shorter
 * than it was written, silently. */
char *engram_rbuf_str(engram_rbuf *r, size_t maxlen);

/* A length-prefixed blob: returns a VIEW (not a copy) and its length, bounded by maxlen. */
const uint8_t *engram_rbuf_blob(engram_rbuf *r, size_t maxlen, size_t *len);

size_t    engram_rbuf_left(const engram_rbuf *r);
size_t    engram_rbuf_pos(const engram_rbuf *r);
engram_rc engram_rbuf_status(const engram_rbuf *r);

/* Record a SEMANTIC failure found by the caller (a count that is impossible, a magic that is wrong),
 * so the whole parse reports one first-cause error. Only the first error is kept. */
void engram_rbuf_fail(engram_rbuf *r, engram_rc rc);

/* THE PARSE IS NOT FINISHED UNTIL THIS SAYS SO: the sticky error if there is one, else
 * ENGRAM_E_FORMAT if any byte remains unconsumed, else ENGRAM_OK. */
engram_rc engram_rbuf_end(const engram_rbuf *r);

/* ---- RAW LITTLE-ENDIAN HELPERS (for fixed headers built in place) ----------------------------- */
void     engram_le_put_u32(uint8_t *p, uint32_t v);
void     engram_le_put_u64(uint8_t *p, uint64_t v);
uint32_t engram_le_get_u32(const uint8_t *p);
uint64_t engram_le_get_u64(const uint8_t *p);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_BUF_H */
