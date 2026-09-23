/* solo/e2e.c -- the store END TO END, through its API: whole paragraphs in (the store cuts them), cues
 * cut from the paragraphs at arbitrary word starts (so a cue may straddle two episodes), recall with
 * k = C to see every candidate, and the three rankings compared on the SAME candidates.
 *   fragment arms  24 / 40 / 64 / 96 bytes x 0 / 1 / 2 typos. HIT: the first episode covers at least
 *                  half of the cue's bytes (in the paragraph it was cut from)
 *   control arms   kw3 / kw5 / kw8 / shuf pieces of one episode (see rarity4.c). HIT: the first episode
 *                  holds every piece
 * usage: e2e docs.txt overlap nsrc dev|test [C] [4] */
#include "engram_store.h"
#include "engram_chunk.h"
#include "engram_alloc.h"
#include "engram_plat.h"
#include "engram_rng.h"
#include "engram_text.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { size_t doc, off, len; } span;
static const char **D; static size_t *DL, ND;
static span *SP; static size_t NSP;                /* by id - 1 */

static int is_letter(uint32_t cp) { unsigned k = engram_cp_class(cp); return k == ENGRAM_CLASS_WORD || k == ENGRAM_CLASS_IDEO; }

/* cue lengths: the default six, or with a 7th argument "4" the first protocol's four (24..96 bytes),
 * which produced results/e2e_dev_ov*.txt */
#define MAX_LEN_ARMS 6
static size_t LEN_ARMS = 6;
static const size_t LENS6[6] = { 12, 16, 24, 40, 64, 96 }, LENS4[4] = { 24, 40, 64, 96 };
static const size_t *LENS = LENS6;
#define NF_MAX 18
static size_t NF = 18;
#define NC 4
static const char *CN[NC] = { "kw3", "kw5", "kw8", "shuf" };

static size_t decode(const char *s, size_t n, uint32_t *cp, size_t *at)
{
    size_t o = 0, k = 0;
    while (o < n) { int v; if (at) at[k] = o; o += engram_utf8_decode((const uint8_t *)s + o, n - o, &cp[k], &v); k++; }
    if (at) at[k] = n;
    return k;
}
static size_t encode(const uint32_t *cp, size_t n, char *out)
{
    size_t i, k = 0;
    for (i = 0; i < n; i++) k += engram_utf8_encode(cp[i], (uint8_t *)out + k);
    return k;
}

/* the three rankings over the same candidates; returns the id each puts first */
static void firsts(const engram_hit *h, size_t n, uint64_t out[3])
{
    size_t i, b = 0, e = 0;
    for (i = 1; i < n; i++) {
        if (h[i].contain > h[b].contain || (h[i].contain == h[b].contain &&
            (h[i].exact > h[b].exact || (h[i].exact == h[b].exact && h[i].id < h[b].id)))) b = i;
        if (h[i].exact > h[e].exact || (h[i].exact == h[e].exact && h[i].id < h[e].id)) e = i;
    }
    out[0] = h[0].id; out[1] = h[b].id; out[2] = h[e].id;
}

int main(int argc, char **argv)
{
    static char buf[64 << 10];
    static uint32_t cp[64 << 10], out[64 << 10];
    static size_t at[64 << 10];
    engram_store_cfg cfg;
    engram_store *s;
    uint8_t *raw; size_t rawn, i, ov, nsrc, t, got = 0, C;
    int test;
    engram_hit *h;
    double fr[NF_MAX][3], fc[NC][3], candf[NF_MAX], candc[NC];
    unsigned pwin[2] = { 0, 0 }, plos[2] = { 0, 0 };   /* FULL vs BAG wins / losses: frag, ctrl */
    unsigned nfa[NF_MAX], nca[NC];
    double t0;
    if (argc < 5) return 2;
    ov = (size_t)atoll(argv[2]); nsrc = (size_t)atoll(argv[3]); test = strcmp(argv[4], "test") == 0;
    C = argc > 5 ? (size_t)atoll(argv[5]) : 50u;
    if (argc > 6 && strcmp(argv[6], "4") == 0) { LEN_ARMS = 4; LENS = LENS4; NF = 12; }
    memset(fr, 0, sizeof fr); memset(fc, 0, sizeof fc); memset(nfa, 0, sizeof nfa); memset(nca, 0, sizeof nca);
    memset(candf, 0, sizeof candf); memset(candc, 0, sizeof candc);
    if (engram_file_read(argv[1], &raw, &rawn) != ENGRAM_OK) return 2;
    D = (const char **)calloc(rawn / 8 + 1, sizeof *D); DL = (size_t *)calloc(rawn / 8 + 1, sizeof *DL);
    for (i = 0; i < rawn; ) { size_t e = i; while (e < rawn && raw[e] != '\n') e++; if (e > i) { D[ND] = (const char *)raw + i; DL[ND++] = e - i; } i = e + 1; }
    engram_store_cfg_default(&cfg);
    cfg.max_episodes = 1000000u; cfg.max_text_bytes = (size_t)256 << 20; cfg.chunk_overlap = ov; cfg.recall_c = (unsigned)C;
    if (engram_store_open(&s, &cfg) != ENGRAM_OK) return 2;
    SP = (span *)calloc(4 * ND + 1000000u, sizeof *SP);
    t0 = (double)engram_now_ns();
    for (i = 0; i < ND; i++) {
        uint64_t first; size_t nadd, off, len, k = 0;
        engram_chunkit it;
        uint64_t sg[ENGRAM_SIG_WORDS];
        if (engram_store_add(s, D[i], DL[i], i, 0u, &first, &nadd) != ENGRAM_OK) continue;
        engram_chunkit_init(&it, D[i], DL[i], cfg.chunk_cap, cfg.chunk_overlap);
        while (engram_chunkit_next(&it, &off, &len)) {
            if (engram_encode_sig(&cfg.enc, D[i] + off, len, sg, NULL) != ENGRAM_OK) continue;
            SP[first - 1 + k].doc = i; SP[first - 1 + k].off = off; SP[first - 1 + k].len = len; k++;
        }
        if (k != nadd) { fprintf(stderr, "span mismatch doc %lu\n", (unsigned long)i); return 3; }
        NSP = first - 1 + k;
    }
    {
        engram_store_stats st; engram_store_stats_get(s, &st);
        fprintf(stderr, "overlap %lu: %lu docs -> %lu episodes, %lu text bytes, built in %.1f s\n", (unsigned long)ov,
                (unsigned long)ND, (unsigned long)st.live, (unsigned long)st.text_bytes, ((double)engram_now_ns() - t0) / 1e9);
    }
    h = (engram_hit *)calloc(C, sizeof *h);
    t0 = (double)engram_now_ns();
    for (t = (size_t)test; got < nsrc; t += 2u) {
        size_t d = (t * 7919u + 13u) % ND, n, a, ns = 0;
        static size_t starts[64 << 10];
        uint32_t al[4096]; size_t nal = 0;
        if (DL[d] < 120u || DL[d] > sizeof buf / 2) continue;
        got++;
        n = decode(D[d], DL[d], cp, at);
        for (i = 0; i < n; i++) {
            if (is_letter(cp[i]) && nal < 4096u) al[nal++] = cp[i];
            if (engram_cp_class(cp[i]) == ENGRAM_CLASS_IDEO || (is_letter(cp[i]) && (i == 0 || engram_cp_class(cp[i - 1]) == ENGRAM_CLASS_SPACE)))
                starts[ns++] = i;
        }
        if (!nal) { al[0] = 'e'; nal = 1; }
        /* ---- fragment arms ---- */
        for (a = 0; a < NF; a++) {
            engram_rng r; size_t L = LENS[a % LEN_ARMS], kk = a / LEN_ARMS, st0, en, m, x, qn, tl, nh = 0, cands, k2;
            size_t pool = 0; uint64_t f[3]; unsigned rk;
            engram_rng_seed(&r, engram_mix2(0xE2E0u + a, (uint64_t)d));
            for (x = 0; x < ns; x++) if (at[starts[x]] + L <= DL[d]) pool++;
            if (!pool) continue;
            k2 = (size_t)engram_rng_below(&r, pool);
            for (x = 0; x < ns; x++) if (at[starts[x]] + L <= DL[d]) { if (!k2) break; k2--; }
            st0 = starts[x];
            for (en = st0; en < n && at[en + 1] - at[st0] <= L; en++) ;
            m = en - st0; memcpy(out, cp + st0, m * sizeof *out); qn = m;
            for (k2 = 0; k2 < kk; k2++) {                      /* typos, as the lab's v_typos */
                size_t p = 0, tries; unsigned op;
                for (tries = 0; tries < 64u; tries++) { p = (size_t)engram_rng_below(&r, qn); if (is_letter(out[p])) break; }
                if (!is_letter(out[p])) break;
                op = (unsigned)engram_rng_below(&r, 4u);
                if (op == 0u) out[p] = al[engram_rng_below(&r, nal)];
                else if (op == 1u && qn > 1u) { memmove(out + p, out + p + 1, (qn - p - 1u) * sizeof *out); qn--; }
                else if (op == 2u) { memmove(out + p + 1, out + p, (qn - p) * sizeof *out); out[p] = al[engram_rng_below(&r, nal)]; qn++; }
                else if (p + 1u < qn) { uint32_t tmp = out[p]; out[p] = out[p + 1]; out[p + 1] = tmp; }
            }
            tl = encode(out, qn, buf);
            if (engram_store_recall(s, buf, tl, h, C, &nh) != ENGRAM_OK || !nh) { nfa[a]++; continue; }
            nfa[a]++;
            firsts(h, nh, f);
            {
                size_t cs = at[st0], ce = at[en];
                int okr[3];
                for (rk = 0; rk < 3u; rk++) {
                    span e = SP[f[rk] - 1];
                    size_t lo = e.off > cs ? e.off : cs, hi = e.off + e.len < ce ? e.off + e.len : ce;
                    okr[rk] = e.doc == d && hi > lo && 2u * (hi - lo) >= ce - cs;
                    if (okr[rk]) fr[a][rk] += 1.0;
                }
                pwin[0] += okr[0] && !okr[1]; plos[0] += !okr[0] && okr[1];
                for (cands = 0, x = 0; x < nh; x++) {
                    span e = SP[h[x].id - 1];
                    size_t lo = e.off > cs ? e.off : cs, hi = e.off + e.len < ce ? e.off + e.len : ce;
                    if (e.doc == d && hi > lo && 2u * (hi - lo) >= ce - cs) { cands = 1; break; }
                }
                candf[a] += (double)cands;
            }
        }
        /* ---- control arms: pieces of one episode of this paragraph ---- */
        {
            size_t e0 = 0, e1 = 0, x;
            for (x = 0; x < NSP; x++) if (SP[x].doc == d) { if (!e1) e0 = x; e1 = x + 1; }
            if (!e1) continue;
            for (a = 0; a < NC; a++) {
                engram_rng r; size_t ep, pn = 0, ps[4096], pl[4096], perm[4096], lo = 0, hi, want, y, m = 0, tl, nh = 0;
                uint64_t f[3]; unsigned rk;
                const char *et; size_t el, en2;
                engram_rng_seed(&r, engram_mix2(0xC7A1u + a, (uint64_t)d));
                ep = e0 + (size_t)engram_rng_below(&r, e1 - e0);
                et = D[d] + SP[ep].off; el = SP[ep].len;
                en2 = decode(et, el, cp, at);
                for (x = 0; x < en2 && pn < 4096u; ) {
                    unsigned cl = engram_cp_class(cp[x]); size_t s0 = x;
                    if (cl == ENGRAM_CLASS_IDEO) {
                        while (x < en2 && x - s0 < 2u && engram_cp_class(cp[x]) == ENGRAM_CLASS_IDEO) x++;
                        if (x - s0 == 2u) { ps[pn] = s0; pl[pn] = 2; pn++; }
                        continue;
                    }
                    if (cl == ENGRAM_CLASS_WORD) {
                        while (x < en2 && engram_cp_class(cp[x]) == ENGRAM_CLASS_WORD) x++;
                        if (x - s0 >= 3u) { ps[pn] = s0; pl[pn] = x - s0; pn++; }
                        continue;
                    }
                    x++;
                }
                if (pn < 2u) continue;
                hi = pn;
                if (a == 3u) {
                    size_t sp = (pn * 15u + 99u) / 100u; if (sp < 2u) sp = 2u; if (sp > pn) sp = pn;
                    lo = pn > sp ? (size_t)engram_rng_below(&r, pn - sp + 1u) : 0u; hi = lo + sp; want = sp;
                } else want = a == 0u ? 3u : a == 1u ? 5u : 8u;
                for (x = lo; x < hi; x++) perm[x - lo] = x;
                for (x = hi - lo; x > 1u; x--) { size_t z = (size_t)engram_rng_below(&r, x), tmp = perm[x - 1]; perm[x - 1] = perm[z]; perm[z] = tmp; }
                if (want > hi - lo) want = hi - lo;
                for (x = 0; x < want; x++) { if (m) out[m++] = ' '; for (y = 0; y < pl[perm[x]]; y++) out[m++] = cp[ps[perm[x]] + y]; }
                tl = encode(out, m, buf);
                if (engram_store_recall(s, buf, tl, h, C, &nh) != ENGRAM_OK || !nh) { nca[a]++; continue; }
                nca[a]++;
                firsts(h, nh, f);
                int okc[3];
                for (rk = 0; rk < 3u; rk++) {
                    span e = SP[f[rk] - 1]; int all = e.doc == d;
                    for (x = 0; x < want && all; x++) {
                        size_t bs = SP[ep].off + at[ps[perm[x]]], be = SP[ep].off + at[ps[perm[x]] + pl[perm[x]]];
                        if (bs < e.off || be > e.off + e.len) all = 0;
                    }
                    fc[a][rk] += all; okc[rk] = all;
                }
                pwin[1] += okc[0] && !okc[1]; plos[1] += !okc[0] && okc[1];
                for (x = 0; x < nh; x++) if (h[x].id - 1 == ep) { candc[a] += 1.0; break; }
            }
        }
    }
    fprintf(stderr, "recall: %.2f ms per cue\n", ((double)engram_now_ns() - t0) / 1e6 / (double)(got * (NF + NC)));
    {
        double sf[3] = { 0 }, sc[3] = { 0 }, cf = 0, cc = 0; unsigned nf = 0, nc = 0, a, rk;
        printf("overlap %lu %s C=%lu nsrc %lu\n", (unsigned long)ov, test ? "test" : "dev", (unsigned long)C, (unsigned long)nsrc);
        for (a = 0; a < NF; a++) {
            printf("  frag %2luB/%lu typos  n=%u  FULL %.4f  BAG %.4f  EXACT %.4f  cand %.4f\n", (unsigned long)LENS[a % LEN_ARMS], (unsigned long)(a / LEN_ARMS),
                   nfa[a], fr[a][0] / nfa[a], fr[a][1] / nfa[a], fr[a][2] / nfa[a], candf[a] / nfa[a]);
            for (rk = 0; rk < 3u; rk++) sf[rk] += fr[a][rk];
            nf += nfa[a]; cf += candf[a];
        }
        for (a = 0; a < NC; a++) {
            printf("  ctrl %-4s        n=%u  FULL %.4f  BAG %.4f  EXACT %.4f  cand %.4f\n", CN[a], nca[a], fc[a][0] / nca[a], fc[a][1] / nca[a], fc[a][2] / nca[a], candc[a] / nca[a]);
            for (rk = 0; rk < 3u; rk++) sc[rk] += fc[a][rk];
            nc += nca[a]; cc += candc[a];
        }
        printf("  FRAGMENTS  FULL %.4f  BAG %.4f  EXACT %.4f  (candidate recall %.4f, n=%u)\n", sf[0] / nf, sf[1] / nf, sf[2] / nf, cf / nf, nf);
        printf("  CONTROL    FULL %.4f  BAG %.4f  EXACT %.4f  (candidate recall %.4f, n=%u)\n", sc[0] / nc, sc[1] / nc, sc[2] / nc, cc / nc, nc);
        printf("  FULL vs BAG paired: fragments +%u -%u, control +%u -%u\n", pwin[0], plos[0], pwin[1], plos[1]);
    }
    engram_store_close(s);
    return 0;
}
