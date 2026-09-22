#!/usr/bin/env python3
# ref_core.py -- INDEPENDENT reference implementations of engram_mix64, engram_mix2, engram_hash_bytes
# and xoshiro256** (seeded by splitmix64). test/test_core.c pins its expected values to what THIS file
# prints, so the C is checked against a second implementation rather than against itself.
# mix64(0) = 0xE220A8397B1DCDAF also matches the published splitmix64 first output for seed 0.
M=(1<<64)-1
def mix64(x):
    x=(x+0x9E3779B97F4A7C15)&M
    x=((x^(x>>30))*0xBF58476D1CE4E5B9)&M
    x=((x^(x>>27))*0x94D049BB133111EB)&M
    return x^(x>>31)
def hash_bytes(b,seed):
    h=0xCBF29CE484222325 ^ mix64(seed)
    for c in b:
        h^=c; h=(h*0x100000001B3)&M
    return mix64(h ^ len(b))
def mix2(a,b):
    return mix64(mix64(a) ^ ((b + 0x632BE59BD9B4E019 + ((a<<6)&M) + (a>>2))&M))
def splitmix_next(st):
    st[0]=(st[0]+0x9E3779B97F4A7C15)&M; z=st[0]
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&M
    z=((z^(z>>27))*0x94D049BB133111EB)&M
    return z^(z>>31)
def rotl(x,k): return ((x<<k)|(x>>(64-k)))&M
class X:
    def __init__(s,seed):
        st=[seed]; s.s=[splitmix_next(st) for _ in range(4)]
    def next(s):
        r=rotl((s.s[1]*5)&M,7)*9&M
        t=(s.s[1]<<17)&M
        s.s[2]^=s.s[0]; s.s[3]^=s.s[1]; s.s[1]^=s.s[2]; s.s[0]^=s.s[3]
        s.s[2]^=t; s.s[3]=rotl(s.s[3],45)
        return r
for v in [0,1,M,0x0123456789ABCDEF]: print("mix64 %#018x -> 0x%016XULL"%(v,mix64(v)))
for b,sd in [(b"",0),(b"abc",0),(b"abc",1),(b"a\x00",0)]: print("hash %r seed %d -> 0x%016XULL"%(b,sd,hash_bytes(b,sd)))
print("mix2(1,2) -> 0x%016XULL  mix2(2,1) -> 0x%016XULL"%(mix2(1,2),mix2(2,1)))
for sd in [0,12345]:
    x=X(sd); print("xoshiro seed %d -> "%sd + ", ".join("0x%016XULL"%x.next() for _ in range(3)))
