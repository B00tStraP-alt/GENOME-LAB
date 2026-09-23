import sys, collections, statistics
split = sys.argv[1] if len(sys.argv) > 1 else 'dev'
C = int(sys.argv[2]) if len(sys.argv) > 2 else 50
def load(fam):
    Q = collections.OrderedDict()
    for l in open('dump_%s_%s.tsv' % (fam, split)):
        p = l.split('\t')
        if p[2].startswith('FAIL'): Q.setdefault(int(p[0]), (int(p[1]), [])); continue
        t, arm, j, src, s1, i50, i200, cont, bh, ed, qn, dn = p
        Q.setdefault(int(t), (int(arm), []))[1].append(dict(src=int(src), s1=float(s1), i50=int(i50), i200=int(i200),
            cont=float(cont), bh=float(bh), ed=int(ed), qn=int(qn), dn=int(dn)))
    return Q
FAM = {f: load(f) for f in ('frag', 'ctrl')}
def cand(rows):
    k = 'i50' if C == 50 else 'i200'
    return [r for r in rows if r[k] or r['src']]

def sa(r): return max(0.0, 1.0 - r['ed'] / r['qn']) if r['qn'] else 0.0
RULES = collections.OrderedDict()
RULES['bhatt'] = lambda r, ctx: (r['bh'],)
RULES['lex(cont,bh)'] = lambda r, ctx: (r['cont'], r['bh'])
RULES['lex(max(sa,cont),bh)'] = lambda r, ctx: (max(sa(r), r['cont']), r['bh'])
RULES['lex(max(sa,cont),cont,bh)'] = lambda r, ctx: (max(sa(r), r['cont']), r['cont'], r['bh'])
RULES['lex(max(sa,cont),sa,bh)'] = lambda r, ctx: (max(sa(r), r['cont']), sa(r), r['bh'])
RULES['lex(sa+cont,bh)'] = lambda r, ctx: (sa(r) + r['cont'], r['bh'])
for div in (2, 3, 4):
    for frac in (3, 4, 5):
        RULES['gate ed<=med/%d & ed<=q/%d' % (div, frac)] = (lambda div, frac: lambda r, ctx:
            ((-r['ed'],) if ctx['gate'][(div, frac)] else ()) + (r['cont'], r['bh']))(div, frac)

def evaluate(fam):
    res = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0]))
    for t, (arm, rows) in FAM[fam].items():
        cs = cand(rows)
        srcs = [r for r in cs if r['src']]
        inC = srcs and (srcs[0]['i50'] if C == 50 else srcs[0]['i200'])
        ctx = {}
        if cs:
            eds = sorted(r['ed'] for r in cs)
            med = statistics.median(eds); best = eds[0]; qn = cs[0]['qn']
            ctx['gate'] = {(d, f): (best * d <= med and best * f <= qn) for d in (2, 3, 4) for f in (3, 4, 5)}
        for name, fn in RULES.items():
            cell = res[name][arm]; cell[2] += 1
            if not inC: continue
            s = fn(srcs[0], ctx)
            above = ties = 0
            for r in cs:
                if r['src']: continue
                x = fn(r, ctx)
                if x > s: above += 1
                elif x == s: ties += 1
            if not above:
                cell[0] += 1.0 / (1 + ties); cell[1] += (ties == 0)
    return res
for fam in ('frag', 'ctrl'):
    res = evaluate(fam)
    arms = sorted(next(iter(res.values())).keys())
    print("== %s (%s, C=%d): expected r@1 overall | per arm" % (fam, split, C))
    for name in RULES:
        tot = sum(res[name][a][0] for a in arms) / sum(res[name][a][2] for a in arms)
        pes = sum(res[name][a][1] for a in arms) / sum(res[name][a][2] for a in arms)
        print("  %-30s %.4f (pess %.4f) |" % (name, tot, pes), ' '.join('%.3f' % (res[name][a][0] / res[name][a][2]) for a in arms))
