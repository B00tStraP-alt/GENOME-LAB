#!/bin/bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while read arm ds lr; do for seed in 1 2 3; do
  echo "SAVE=runs/model_${arm}_${ds}_${seed}.npz python3 train.py $arm $ds $lr 10 $seed test > runs/${arm}_${ds}_${lr}_${seed}_test.txt 2>&1"
done; done < "$1" | xargs -P 4 -I{} bash -c "{}"
