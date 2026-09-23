/* P1.3 opening experiment: stage-1 recall@C and cascade r@1 as the store grows. argv: N nsrc */
#define main test_enc_main_unused
#include "../../test/test_enc.c"
#undef main
static const unsigned CS[] = { 10, 50, 100, 200, 500 };
#define NC 5
int main(int argc, char **argv)
{
    corpus all, c; engram_enc_cfg cfg; size_t N = (size_t)atoll(argv[1]), nsrc = (size_t)atoll(argv[2]), t, j;
    static uint32_t in[MAXCP], out[MAXCP]; static char txt[MAXB]; static float v[ENGRAM_D];
    float *mat; uint64_t *sig; engram_encq *eq = engram_malloc(sizeof *eq), *dt = engram_malloc(sizeof *dt);
    double *ss, *sd, *srt; unsigned rs[NC] = {0}, rd[NC] = {0}, cas[NC] = {0}, nq = 0, hx = 0;
    double t0, tsig = 0, tdense = 0; size_t *src; char **qt; size_t *ql;
    corpus_load(&all, "scale"); c = all; if (N < c.n) c.n = N;
    engram_enc_cfg_default(&cfg);
    mat = index_build(&cfg, ENGRAM_D, &c); sig = sig_build(&cfg, &c);
    ss = engram_array(c.n, 8); sd = engram_array(c.n, 8); srt = engram_array(c.n, 8);
    src = engram_array(nsrc * NARM, sizeof *src); qt = engram_array(nsrc * NARM, sizeof *qt); ql = engram_array(nsrc * NARM, sizeof *ql);
    /* queries: nsrc sources spread through the store, 12 arms each, typos from the source's own letters */
    for (t = 0; t < nsrc; t++) {
        size_t i = (t * 7919u) % c.n, n = decode_all(c.line[i], c.len[i], in, MAXCP), k, a; alphabet ab; static uint32_t al[MAXCP]; ab.a = al; ab.n = 0;
        for (k = 0; k < n; k++) if (is_letter(in[k])) al[ab.n++] = in[k];
        if (!ab.n) { al[0] = 'e'; ab.n = 1; }
        for (a = 0; a < NARM; a++) { ft_arg fa; engram_rng r; size_t m, tl; fa.pct = ARM_PCT[a % 4]; fa.k = (unsigned)(a / 4); fa.ab = &ab;
            engram_rng_seed(&r, engram_mix2(0x5CA1Eu + a, i)); m = v_fragtypo(&r, in, n, out, MAXCP, &fa); tl = encode_all(out, m, txt, sizeof txt);
            src[nq] = i; qt[nq] = engram_malloc(tl + 1); memcpy(qt[nq], txt, tl + 1); ql[nq] = tl; nq++; }
    }
    for (t = 0; t < nq; t++) {
        size_t i = src[t]; unsigned k, above_s = 0, above_d = 0; double e_src, e;
        if (engram_encq_build(eq, &cfg, qt[t], ql[t]) != ENGRAM_OK) continue;
        t0 = (double)engram_now_ns();
        for (j = 0; j < c.n; j++) engram_encq_sig_score(eq, sig + j * ENGRAM_SIG_WORDS, &ss[j]);
        tsig += (double)engram_now_ns() - t0;
        engram_encode(&cfg, qt[t], ql[t], v, NULL);
        t0 = (double)engram_now_ns();
        for (j = 0; j < c.n; j++) sd[j] = fdot(v, mat + j * ENGRAM_D, ENGRAM_D);
        tdense += (double)engram_now_ns() - t0;
        for (j = 0; j < c.n; j++) { if (j == i) continue; if (ss[j] > ss[i]) above_s++; if (sd[j] > sd[i]) above_d++; }
        engram_encq_score(eq, c.line[i], c.len[i], &e_src);
        memcpy(srt, ss, c.n * 8); qsort(srt, c.n, 8, double_desc);
        for (k = 0; k < NC; k++) { double thr = srt[(CS[k] < c.n ? CS[k] : c.n) - 1]; int beaten = 0;
            rs[k] += above_s < CS[k]; rd[k] += above_d < CS[k];
            if (ss[i] < thr) continue;
            for (j = 0; j < c.n && !beaten; j++) if (j != i && ss[j] >= thr) { engram_encq_score(eq, c.line[j], c.len[j], &e); if (e > e_src) beaten = 1; }
            cas[k] += !beaten; }
        /* exhaustive exact, only for small stores (by streaming every doc) */
        if (c.n <= 5000) { int beaten = 0; for (j = 0; j < c.n && !beaten; j++) if (j != i) { engram_encq_score(eq, c.line[j], c.len[j], &e); if (e > e_src) beaten = 1; } hx += !beaten; }
    }
    printf("N=%lu queries=%u | sig stage-1 %.1f us/query, dense %.1f us/query\n", (unsigned long)c.n, nq, tsig / nq / 1e3, tdense / nq / 1e3);
    for (unsigned k = 0; k < NC; k++) printf("  C=%-4u stage-1 recall: sig %.4f dense %.4f | cascade(sig) r@1 %.4f\n", CS[k], (double)rs[k] / nq, (double)rd[k] / nq, (double)cas[k] / nq);
    if (c.n <= 5000) printf("  exhaustive exact r@1 %.4f\n", (double)hx / nq);
    return 0;
}
