import sys, statistics
exec(open('evalrules.py').read().split("def evaluate(fam):")[0].replace("split = sys.argv[1] if len(sys.argv) > 1 else 'dev'", "split = sys.argv[1]").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 50", "C = int(sys.argv[2])"))
lines = [l.rstrip('\n') for l in open('corpus_scale.txt', encoding='utf-8', errors='replace')]
def rank1(cs, fn, ctx):
    srcs = [r for r in cs if r['src']]
    if not srcs: return 0.0
    s = fn(srcs[0], ctx); above = ties = 0
    for r in cs:
        if r['src']: continue
        x = fn(r, ctx)
        if x > s: above += 1
        elif x == s: ties += 1
    return 0.0 if above else 1.0 / (1 + ties)
A = RULES['lex(cont,bh)']; B = RULES[sys.argv[3] if len(sys.argv) > 3 else 'gate ed<=med/3 & ed<=q/3']
for fam in ('frag', 'ctrl'):
    win = lose = 0; ex = []
    for t, (arm, rows) in FAM[fam].items():
        cs = cand(rows)
        if not cs: continue
        eds = sorted(r['ed'] for r in cs); med = statistics.median(eds); best = eds[0]; qn = cs[0]['qn']
        ctx = {'gate': {(d, f): (best * d <= med and best * f <= qn) for d in (2, 3, 4) for f in (3, 4, 5)}}
        a = rank1(cs, A, ctx); b = rank1(cs, B, ctx)
        if b > a: win += 1
        if b < a:
            lose += 1
            src = [r for r in cs if r['src']][0]
            top = max(cs, key=lambda r: B(r, ctx))
            ex.append((t, arm, qn, src['ed'], top['ed'], med, src['cont'], top['cont'], top.get('j')))
    print("%s %s C=%d: %s vs lex(cont,bh): wins %d losses %d" % (fam, split, C, sys.argv[3] if len(sys.argv) > 3 else 'gate med/3 q/3', win, lose))
    for e in ex[:6]: print("   loss t=%d arm=%d qn=%d src_ed=%d top_ed=%d med=%.1f src_cont=%.3f top_cont=%.3f" % e[:8])
