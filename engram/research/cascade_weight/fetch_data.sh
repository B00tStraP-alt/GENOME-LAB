#!/usr/bin/env bash
# Downloads MNIST and Fashion-MNIST into ./data and refuses any file whose SHA-256 differs from the
# value recorded when the experiments in README.md were run.
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data; cd data
get() { [ -f "$1" ] || curl -sSfL --max-time 120 -o "$1" "$2"; }
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  get "mnist-$f.gz"   "https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz"
  get "fashion-$f.gz" "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/$f.gz"
done
sha256sum -c <<'SUMS'
346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073  fashion-t10k-images-idx3-ubyte.gz
67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5  fashion-t10k-labels-idx1-ubyte.gz
3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84  fashion-train-images-idx3-ubyte.gz
a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845  fashion-train-labels-idx1-ubyte.gz
8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6  mnist-t10k-images-idx3-ubyte.gz
f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6  mnist-t10k-labels-idx1-ubyte.gz
440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609  mnist-train-images-idx3-ubyte.gz
3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c  mnist-train-labels-idx1-ubyte.gz
SUMS
