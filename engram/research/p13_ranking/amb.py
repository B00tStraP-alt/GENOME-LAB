import sys, collections
split = sys.argv[1] if len(sys.argv) > 1 else 'dev'
lines = [l.rstrip('\n') for l in open('corpus_scale.txt', encoding='utf-8', errors='surrogateescape')]
Q = []
for l in open('queries_%s.tsv' % split, encoding='utf-8', errors='surrogateescape'):
    t, s, q = l.rstrip('\n').split('\t', 2); Q.append((int(t), int(s), q))
R = {}
for l in open('rarity2_q_%s.tsv' % split):
    p = l.rstrip('\n').split('\t')
    if p[2] == 'OUT': R[int(p[0])] = None; continue
    R[int(p[0])] = [tuple(map(int, x.split())) for x in p[2:]]
FN = ['bhatt', 'cont', 'cont-idf', 'lex(cidf,bhatt)']
PCT = [5, 8, 10, 15]
# verbatim ambiguity: other chunks holding the query (0-typo arms) as a substring
amb = {}
for t, s, q in Q:
    a = t % 12
    if a >= 4: continue
    amb[t] = sum(1 for j, L in enumerate(lines) if j != s and q in L)
byarm = collections.defaultdict(lambda: [0, 0])
for t, k in amb.items():
    byarm[PCT[t % 12]][0] += 1; byarm[PCT[t % 12]][1] += (k > 0)
print("0-typo queries whose exact text also occurs in ANOTHER chunk:")
for p in PCT: print("  %2d%%: %d / %d" % (p, byarm[p][1], byarm[p][0]))
ceil = sum(1.0 / (1 + k) for k in amb.values()) / len(amb)
print("  best possible expected r@1 on 0-typo arms for ANY text-only ranker (random among verbatim holders): %.4f" % ceil)
# what each function achieves on 0-typo arms, expected value
for f, name in enumerate(FN):
    e = 0.0; pe = 0.0
    for t in amb:
        r = R[t]
        if r is None: continue
        above, ties = r[f]
        if above == 0: e += 1.0 / (1 + ties); pe += (ties == 0)
    print("  %-18s exp %.4f pess %.4f" % (name, e / len(amb), pe / len(amb)))
# failures of lex(cidf,bhatt) by arm (all arms)
fails = collections.Counter(); tot = collections.Counter()
for t, s, q in Q:
    a = t % 12; tot[a] += 1
    r = R[t]
    if r is None or r[3][0] > 0 or r[3][1] > 0: fails[a] += 1
print("lex(cidf,bhatt) misses by arm (pct, typos): ", ' '.join("%d%%/%d:%d" % (PCT[a % 4], a // 4, fails[a]) for a in range(12)))
fails = collections.Counter()
for t, s, q in Q:
    a = t % 12; r = R[t]
    if r is None or r[0][0] > 0 or r[0][1] > 0: fails[a] += 1
print("bhatt           misses by arm (pct, typos): ", ' '.join("%d%%/%d:%d" % (PCT[a % 4], a // 4, fails[a]) for a in range(12)))
# examples of lex misses on 0-typo arms where the text is NOT verbatim elsewhere (the fixable ones)
n = 0
for t, s, q in Q:
    a = t % 12
    if a >= 4 or amb.get(t, 1) > 0: continue
    r = R[t]
    if r is None or r[3][0] == 0: continue
    print("MISS t=%d arm %d%% src=%d above=%d q=%r" % (t, PCT[a], s, r[3][0], q[:120]))
    n += 1
    if n >= 12: break
