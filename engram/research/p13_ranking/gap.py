import sys, statistics
exec(open('evalrules.py').read().split("def evaluate(fam):")[0].replace("split = sys.argv[1] if len(sys.argv) > 1 else 'dev'", "split = sys.argv[1]").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 50", "C = int(sys.argv[2])"))
# For queries where the med/3,q/3 gate FIRES: where does the aligned winner stand on the bag score?
# gap = max cont among candidates - cont of the best-aligned candidate; and n_at_best = candidates at the best ED
import collections
for fam in ('frag', 'ctrl'):
    good = []; bad = []
    for t, (arm, rows) in FAM[fam].items():
        cs = cand(rows)
        if not cs: continue
        eds = sorted(r['ed'] for r in cs); med = statistics.median(eds); best = eds[0]; qn = cs[0]['qn']
        if not (best * 3 <= med and best * 3 <= qn): continue
        top = max(cs, key=lambda r: (-r['ed'], r['cont'], r['bh']))
        mc = max(r['cont'] for r in cs)
        second = eds[1] if len(eds) > 1 else 99
        (good if top['src'] else bad).append((round(mc - top['cont'], 3), qn, best, second, med))
    print(fam, split, C, "gate fires: winner is source %d, winner is not %d" % (len(good), len(bad)))
    g = sorted(x[0] for x in good)
    print("   bag gap of CORRECT aligned winners: quantiles", [g[int(q * (len(g) - 1))] for q in (0.5, 0.9, 0.95, 0.99, 1.0)] if g else None)
    print("   wrong aligned winners (gap, qn, best, second, med):", bad[:12])
