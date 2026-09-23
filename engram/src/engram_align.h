/* ==================================================================================================
 * engram_align.h -- THE ALIGNMENT: how many edits separate a cue from the best-matching stretch of a
 * text.
 * ==================================================================================================
 *
 * WHY IT EXISTS
 * ==================================================================================================
 * The encoder's features are character n-grams, a bag: they say WHICH pieces of a cue a text holds,
 * not whether it holds them TOGETHER, in order. At 21.8 K episodes that stops being enough. A short,
 * misspelt fragment of one episode shares most of its pieces with dozens of others, and the bag cannot
 * tell the one that contains the fragment from the ones that merely contain its words. An alignment
 * settles it: the source is a few edits from the cue; an impostor is many. Measured (engram_store.h,
 * ledger W-P1.3-9): of the fragments the bag alone ranked wrong, the store's gated alignment recovered
 * 72 of 158 in the lab and 182 of 602 end to end (dev splits), and lost none.
 *
 * WHAT IT COMPUTES
 * ==================================================================================================
 * The semi-global OPTIMAL STRING ALIGNMENT distance (Sellers 1980, with the Damerau transposition):
 * the fewest insertions, deletions, substitutions and swaps of two adjacent codepoints that turn the
 * cue into SOME contiguous stretch of the text (possibly empty). So:
 *   0              the cue occurs in the text verbatim
 *   <= qn          always (delete the whole cue)
 *   >= qn - dn     always (a text shorter than the cue must absorb the difference)
 * Both sides are first put in the encoder's own normalised form -- folded, IGNOREs and folded-away
 * diacritics dropped -- with every run of SPACE collapsed to one U+0020 and the ends trimmed, so the
 * alignment and the bag judge the same characters.
 *
 * Integers only: identical on every platform (R5). Dynamic programming in three rows; O(qn * dn).
 * ============================================================================================== */
#ifndef ENGRAM_ALIGN_H
#define ENGRAM_ALIGN_H

#include "engram.h"
#include "engram_text.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The normalised codepoints of `text`, at most `cap` of them. ENGRAM_E_FULL if there are more (out
 * then holds the first cap and *len == cap); ENGRAM_E_UTF8 if policy is STRICT and the text is
 * ill-formed. A text of n bytes never yields more than ENGRAM_FOLD_OUT_MAX * n codepoints. */
engram_rc engram_align_norm(const void *text, size_t n, engram_fold fold, engram_utf8_policy policy,
                            uint32_t *out, size_t cap, size_t *len);

/* The distance of q[0..qn) to the best stretch of d[0..dn). `work` holds 3 * (dn + 1) entries and is
 * overwritten. Never fails; allocates nothing. */
uint32_t  engram_align_dist(const uint32_t *q, size_t qn, const uint32_t *d, size_t dn, uint32_t *work);

#ifdef __cplusplus
}
#endif

#endif /* ENGRAM_ALIGN_H */
