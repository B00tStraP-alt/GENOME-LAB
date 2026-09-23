/* ==================================================================================================
 * engram_chunk.h -- cutting text into EPISODES: bounded, whole-codepoint, sentence-preferring spans.
 * ==================================================================================================
 *
 * WHY EPISODES ARE BOUNDED
 * ==================================================================================================
 * A signature (engram_enc.h) is a Bloom filter: a text with many thousands of distinct features sets
 * nearly every bit and "contains" every query. And a person recalls a PASSAGE, not a book. So the store
 * holds bounded chunks, and this module decides where one ends and the next begins.
 *
 * THE CUT, IN ORDER OF PREFERENCE
 * ==================================================================================================
 * From the current start, the window is the next `cap` bytes. If everything left fits, it is the last
 * chunk. Otherwise the cut is the LAST of these found in the window's back three quarters:
 *
 *   1. a paragraph break          a newline followed by optional spaces and another newline
 *   2. a sentence end             . ! ? ; or the CJK / fullwidth 。！？；… followed by space, a closing
 *                                 quote or bracket, or the end -- so "3.14" and "e.g." inside a
 *                                 sentence are not cut after the full stop unless a space follows
 *   3. whitespace                 between words
 *   4. any codepoint boundary     a run with no spaces at all (CJK is written without them)
 *
 * The front quarter is excluded so that a stray full stop two words in does not produce a sliver.
 *
 * GUARANTEES -- each one a check in test_chunk.c, over every corpus and over hostile inputs
 * ==================================================================================================
 *   BOUNDED     every chunk is at most `cap` bytes
 *   WHOLE       every chunk begins and ends on a decoder step (engram_utf8_decode), so no codepoint --
 *               and no ill-formed subpart -- is ever split across two chunks
 *   COVERING    every byte that is not whitespace lies in at least one chunk
 *   PROGRESS    each chunk starts strictly after the previous one; the iterator always terminates
 *   NO FLUFF    no chunk is whitespace only; leading and trailing whitespace are trimmed
 *   OVERLAP     with overlap > 0, a chunk begins about `overlap` bytes before the previous one ended,
 *               moved forward to a word start (never before the previous chunk's start)
 *   ALLOCATES   nothing; the chunks are (offset, length) spans into the caller's text
 * ============================================================================================== */
#ifndef ENGRAM_CHUNK_H
#define ENGRAM_CHUNK_H

#include "engram.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ENGRAM_CHUNK_MIN_CAP 32u    /* smaller than this and a sentence rule has no room to act */

typedef struct {
    const uint8_t *p;
    size_t         n;
    size_t         cap, overlap;
    size_t         pos;           /* where the next chunk may begin                     */
    size_t         last_start;    /* the previous chunk's start, for the progress rule  */
    int            started;
    size_t         n_chunks;
    size_t         n_sentence, n_paragraph, n_space, n_hard;   /* how each cut was made */
} engram_chunkit;

/* ENGRAM_E_ARG if it is NULL, text is NULL with n > 0, cap is outside [ENGRAM_CHUNK_MIN_CAP,
 * ENGRAM_EPI_TAIL], or overlap >= cap / 2. */
engram_rc engram_chunkit_init(engram_chunkit *it, const void *text, size_t n, size_t cap, size_t overlap);

/* The next chunk as (offset, length) into the text; returns 0 when there are no more. */
int engram_chunkit_next(engram_chunkit *it, size_t *off, size_t *len);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_CHUNK_H */
