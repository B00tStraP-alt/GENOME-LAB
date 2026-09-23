#!/bin/bash
# run.sh TAG [expert.py args]: one run; its JSON line to results/runs.jsonl, progress to results/TAG.err
cd "$(dirname "$0")"
tag=$1; shift
OPENBLAS_NUM_THREADS=1 nice python3 expert.py --tag "$tag" "$@" 2> "results/$tag.err" >> results/runs.jsonl
