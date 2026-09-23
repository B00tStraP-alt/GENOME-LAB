import numpy as np
from cascade import *
from train import Net, xent
F32=np.float32
# 1. round-trips
rng=np.random.default_rng(0)
p=rng.integers(0,12,size=(37,53)).astype(np.uint8)
assert np.array_equal(unpack4(pack4(p),p.shape),p), "pack4"
v=value_of(p); assert np.array_equal(unpack_trits(pack_trits(v),v.shape),v), "trits"
print("roundtrip ok; pack4 bytes/weight", pack4(p).size/p.size, "trit bytes/weight", pack_trits(v).size/v.size)
# the line: values and depths
pp=np.arange(12,dtype=np.uint8); print("value", value_of(pp).tolist()); print("depth", depth_of(pp).tolist())
# 2. RNG determinism and uniformity
a=uniform24(5,7,100000); b=uniform24(5,7,100000); assert np.array_equal(a,b)
print("rng mean %.4f (0.5) var %.4f (0.0833)"%(a.mean(),a.var()), "differs by step:", not np.array_equal(a,uniform24(5,8,100000)))
# 3. gradient check (fp32 arm, float64 copy for precision)
net=Net("fp32",[20,16,12,5],1,1e-3)
for l in net.layers: l.W=l.W.astype(np.float64); l.alpha=np.float64(l.alpha)
net.gain=[g.astype(np.float64)+rng.standard_normal(g.shape)*0.1 for g in net.gain]; net.bias=net.bias.astype(np.float64)
x=rng.standard_normal((8,20)); y=rng.integers(0,5,8)
def loss_of():
    return xent(net.forward(x),y)[0]
# analytic: intercept the gradient by giving the layer a recording optimiser
grads={}
for i,l in enumerate(net.layers):
    class Rec:
        def __init__(s,i): s.i=i
        def step(s,w,g,sc): grads[s.i]=g.copy()
    l.opt=Rec(i)
for o in net.gopt+[net.bopt]:
    o.step=lambda w,g,sc: None
logits=net.forward(x); L,d=xent(logits,y); net.backward(d,1.0)
worst=0
for i,l in enumerate(net.layers):
    for _ in range(20):
        a_,b_=rng.integers(0,l.W.shape[0]),rng.integers(0,l.W.shape[1])
        eps=1e-6; l.W[a_,b_]+=eps; lp=loss_of(); l.W[a_,b_]-=2*eps; lm=loss_of(); l.W[a_,b_]+=eps
        num=(lp-lm)/(2*eps); ana=grads[i][a_,b_]
        rel=abs(num-ana)/max(1e-8,abs(num)+abs(ana))
        if abs(num-ana)>1e-8: worst=max(worst,rel)      # below 1e-8 absolute is finite-difference noise
print("gradient check worst relative error %.2e"%worst); assert worst<1e-5
# 4. the general word agrees with the 4-bit word, and every K keeps a third of positions per value
assert np.array_equal(value_k(pp, 4), value_of(pp)) and np.array_equal(depth_k(pp, 4), depth_of(pp).astype(np.int32))
for K in (4, 16, 64):
    q = np.arange(3 * K, dtype=np.uint8); v = value_k(q, K); d = depth_k(q, K)
    assert [int((v == x).sum()) for x in (-1, 0, 1)] == [K, K, K] and d.min() == 0 and d.max() == K - 1
print("general word consistent for K = 4, 16, 64")
