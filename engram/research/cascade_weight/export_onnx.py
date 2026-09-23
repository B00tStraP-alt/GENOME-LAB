"""Export a trained Cascade-4 network to a standard ONNX file whose weights are stored as PACKED TRITS
(five per byte, 1.6 bits per weight) and decoded inside the graph with stock operators only --
Cast, Mod, Div, Sub, Reshape, Transpose, MatMul. Any ONNX runtime on any hardware runs it; no custom
kernel exists. Then verify: onnxruntime vs the numpy reference, same inputs."""
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort
from cascade import load, value_k, pack_trits, unpack_trits

F32 = np.float32


def word_k(z):
    arm = str(z["arm"])
    return int(arm[7:]) if arm.startswith("cascade") and len(arm) > 7 else 4


def build(path_npz, path_onnx):
    z = np.load(path_npz, allow_pickle=True)
    K = word_k(z)
    sizes = [int(s) for s in z["sizes"]]
    L = len(sizes) - 1
    nodes, inits = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), name))
        return name

    const("c127", np.array(127.0, F32)); const("eps12", np.array(1e-12, F32)); const("eps6", np.array(1e-6, F32))
    for k in (3, 9, 27, 81):
        const(f"i{k}", np.array(k, np.int32))
    const("i3m", np.array(3, np.int32)); const("i1", np.array(1, np.int32))
    h = "x"
    stored = 0
    for i in range(L):
        v = value_k(z[f"p{i}"], K)                                # (out, in) ternary: the value plane
        packed = pack_trits(v)
        stored += packed.size
        n_out, n_in = v.shape
        const(f"W{i}_trits", packed)                               # uint8, 1.6 bits per weight
        # decode: t_j = (b / 3^j) mod 3, j = 0..4 ; weight = t - 1
        nodes.append(helper.make_node("Cast", [f"W{i}_trits"], [f"b{i}"], to=TensorProto.INT32))
        digits = []
        for j, k in enumerate((1, 3, 9, 27, 81)):
            src = f"b{i}" if k == 1 else f"b{i}_d{j}"
            if k != 1:
                nodes.append(helper.make_node("Div", [f"b{i}", f"i{k}"], [src]))
            nodes.append(helper.make_node("Mod", [src, "i3m"], [f"t{i}_{j}"]))
            digits.append(f"t{i}_{j}")
        for d in digits:
            nodes.append(helper.make_node("Unsqueeze", [d, "ax1"], [f"{d}_u"]))
        nodes.append(helper.make_node("Concat", [f"{d}_u" for d in digits], [f"t{i}"], axis=1))
        const(f"shape{i}", np.array([-1], np.int64))
        nodes.append(helper.make_node("Reshape", [f"t{i}", f"shape{i}"], [f"tf{i}"]))
        const(f"start{i}", np.array([0], np.int64)); const(f"end{i}", np.array([n_out * n_in], np.int64))
        nodes.append(helper.make_node("Slice", [f"tf{i}", f"start{i}", f"end{i}"], [f"ts{i}"]))
        nodes.append(helper.make_node("Sub", [f"ts{i}", "i1"], [f"w{i}_int"]))
        nodes.append(helper.make_node("Cast", [f"w{i}_int"], [f"w{i}_f"], to=TensorProto.FLOAT))
        const(f"wshape{i}", np.array([n_out, n_in], np.int64))
        nodes.append(helper.make_node("Reshape", [f"w{i}_f", f"wshape{i}"], [f"W{i}"]))
        nodes.append(helper.make_node("Transpose", [f"W{i}"], [f"WT{i}"], perm=[1, 0]))
        # int8 activation quantisation, per sample: s = max|h| / 127 ; hq = round(h / s)
        nodes.append(helper.make_node("Abs", [h], [f"a{i}"]))
        nodes.append(helper.make_node("ReduceMax", [f"a{i}", "ax1_64"], [f"m{i}"], keepdims=1))
        nodes.append(helper.make_node("Div", [f"m{i}", "c127"], [f"s0{i}"]))
        nodes.append(helper.make_node("Add", [f"s0{i}", "eps12"], [f"s{i}"]))
        nodes.append(helper.make_node("Div", [h, f"s{i}"], [f"hd{i}"]))
        nodes.append(helper.make_node("Round", [f"hd{i}"], [f"hq{i}"]))
        nodes.append(helper.make_node("Mul", [f"hq{i}", f"s{i}"], [f"hh{i}"]))
        nodes.append(helper.make_node("MatMul", [f"hh{i}", f"WT{i}"], [f"mm{i}"]))
        const(f"alpha{i}", np.array(z["alphas"][i], F32))
        nodes.append(helper.make_node("Mul", [f"mm{i}", f"alpha{i}"], [f"z{i}"]))
        # RMSNorm with gain
        nodes.append(helper.make_node("Mul", [f"z{i}", f"z{i}"], [f"zz{i}"]))
        nodes.append(helper.make_node("ReduceMean", [f"zz{i}", "ax1_64"], [f"ms{i}"], keepdims=1))
        nodes.append(helper.make_node("Add", [f"ms{i}", "eps6"], [f"mse{i}"]))
        nodes.append(helper.make_node("Sqrt", [f"mse{i}"], [f"r{i}"]))
        nodes.append(helper.make_node("Div", [f"z{i}", f"r{i}"], [f"n{i}"]))
        const(f"g{i}", z["gains"][i].astype(F32))
        nodes.append(helper.make_node("Mul", [f"n{i}", f"g{i}"], [f"y{i}"]))
        if i < L - 1:
            nodes.append(helper.make_node("Relu", [f"y{i}"], [f"h{i}"]))
            h = f"h{i}"
        else:
            const("bias", z["bias"].astype(F32))
            nodes.append(helper.make_node("Add", [f"y{i}", "bias"], ["logits"]))
    const("ax1", np.array([1], np.int64)); const("ax1_64", np.array([1], np.int64))
    graph = helper.make_graph(nodes, "cascade4", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, sizes[0]])],
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, sizes[-1]])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path_onnx)
    nweights = sum(int(np.prod(z[f"p{i}"].shape)) for i in range(L))
    return nweights, stored


def numpy_forward(path_npz, x):
    z = np.load(path_npz, allow_pickle=True)
    K = word_k(z)
    L = len(z["sizes"]) - 1
    h = x
    for i in range(L):
        W = value_k(z[f"p{i}"], K).astype(F32)
        s = np.max(np.abs(h), axis=1, keepdims=True) / F32(127.0) + F32(1e-12)
        hh = np.rint(h / s) * s
        zz = (hh @ W.T) * F32(z["alphas"][i])
        r = np.sqrt(np.mean(zz * zz, axis=1, keepdims=True) + F32(1e-6))
        y = zz / r * z["gains"][i].astype(F32)
        h = np.maximum(y, 0) if i < L - 1 else y + z["bias"].astype(F32)
    return h


if __name__ == "__main__":
    npz, out = sys.argv[1], sys.argv[2]
    nweights, stored = build(npz, out)
    import os
    print(f"exported {out}: {nweights} weights stored in {stored} bytes = {8 * stored / nweights:.3f} bits/weight;"
          f" file {os.path.getsize(out)} bytes")
    _, _, xte, yte = load(sys.argv[3] if len(sys.argv) > 3 else "mnist", "data")
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    lo = sess.run(["logits"], {"x": xte})[0]
    ln = numpy_forward(npz, xte)
    agree = float(np.mean(lo.argmax(1) == ln.argmax(1)))
    print(f"onnxruntime accuracy {np.mean(lo.argmax(1) == yte):.4f}  numpy accuracy {np.mean(ln.argmax(1) == yte):.4f}"
          f"  prediction agreement {agree:.4f}  max |logit diff| {np.max(np.abs(lo - ln)):.2e}")
