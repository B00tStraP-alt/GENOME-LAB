import re, sys
MAXB = 480
def body(t):
    s = re.search(r'\*\*\* ?START OF[^\n]*\n', t); e = re.search(r'\*\*\* ?END OF', t)
    return t[s.end() if s else 0 : e.start() if e else len(t)]
def paras(t, zh):
    t = t.replace('\r', '')
    if zh:
        out, cur = [], ''
        for line in t.split('\n'):
            if line.startswith('　') or not line.strip():
                if cur.strip(): out.append(cur.strip())
                cur = line.strip()
            else: cur += line.strip()
        if cur.strip(): out.append(cur.strip())
        return out
    return [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
def split_sent(p, zh):
    return re.split(r'(?<=[。！？])' if zh else r'(?<=[.!?;:])\s+', p)
def chunks(ps, zh):
    out, cur = [], ''
    sep = '' if zh else ' '
    for p in ps:
        for s in split_sent(p, zh):
            s = s.strip()
            if not s: continue
            while len(s.encode()) > MAXB:           # a single over-long sentence: hard cut on a char
                b = s.encode()[:MAXB].decode('utf-8', 'ignore'); out.append(b); s = s[len(b):]
            cand = (cur + sep + s) if cur else s
            if len(cand.encode()) <= MAXB: cur = cand
            else:
                if cur: out.append(cur)
                cur = s
        if cur and len(cur.encode()) >= 200: out.append(cur); cur = ''   # paragraph end closes a full chunk
    if cur: out.append(cur)
    return [c for c in out if len(c.encode()) >= 40]
allc = []
for f, zh in [('pg11.txt',0),('pg1342.txt',0),('pg1661.txt',0),('pg84.txt',0),('pg14155.txt',0),
              ('pg13371.txt',0),('pg2229.txt',0),('pg22367.txt',0),('pg23950.txt',1),('pg24264.txt',1)]:
    t = body(open(f, encoding='utf-8', errors='replace').read())
    c = chunks(paras(t, zh), zh); allc += c
    print(f, len(c), file=sys.stderr)
import random
random.seed(7); random.shuffle(allc)            # interleave languages so any prefix is a mixed store
open('corpus_scale.txt', 'w', encoding='utf-8').write('\n'.join(allc) + '\n')
L = sorted(len(c.encode()) for c in allc)
print('total', len(allc), 'bytes median', L[len(L)//2], 'max', L[-1], file=sys.stderr)
