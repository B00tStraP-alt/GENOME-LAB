import sys, statistics
exec(open('evalrules.py').read().split("def evaluate(fam):")[0].replace("split = sys.argv[1] if len(sys.argv) > 1 else 'dev'", "split = sys.argv[1]").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 50", "C = int(sys.argv[2])"))
def rank1(cs, key):
    srcs = [r for r in cs if r['src']]
    if not srcs: return 0.0
    s = key(srcs[0]); above = ties = 0
    for r in cs:
        if r['src']: continue
        x = key(r)
        if x > s: above += 1
        elif x == s: ties += 1
    return 0.0 if above else 1.0 / (1 + ties)
# per-candidate key: aligned candidates (significant ED, and bag within delta of the best bag) first by ED
def mk(delta, div=3, frac=3):
    def prep(cs):
        eds = sorted(r['ed'] for r in cs); med = statistics.median(eds); mc = max(r['cont'] for r in cs)
        def key(r):
            ok = r['ed'] * div <= med and r['ed'] * frac <= r['qn'] and r['cont'] >= mc - delta
            return (1, -r['ed'], r['cont'], r['bh']) if ok else (0, 0, r['cont'], r['bh'])
        return key
    return prep
base = lambda cs: (lambda r: (0, 0, r['cont'], r['bh']))
V = [('lex(cont,bh)', base)] + [('aligned, delta=%s' % d, mk(d)) for d in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 9)]
for fam in ('frag', 'ctrl'):
    B = []
    for t, (arm, rows) in FAM[fam].items():
        cs = cand(rows)
        B.append([rank1(cs, p(cs)) if cs else 0.0 for _, p in V])
    n = len(B)
    for k, (name, _) in enumerate(V):
        w = sum(1 for b in B if b[k] > b[0]); l = sum(1 for b in B if b[k] < b[0])
        print("%s %s C=%d %-20s r@1 %.4f  vs bag: +%d -%d" % (fam, split, C, name, sum(b[k] for b in B) / n, w, l))
