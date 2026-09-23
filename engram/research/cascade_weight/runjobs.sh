#!/bin/bash
# runjobs.sh JOBFILE SPLIT EPOCHS SEED -> runs/<arm>_<ds>_<lr>_<seed>_<split>.txt, 4 at a time, 1 BLAS thread each
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while read arm ds lr; do
  echo "python3 train.py $arm $ds $lr $3 $4 $2 > runs/${arm}_${ds}_${lr}_$4_$2.txt 2>&1"
done < "$1" | xargs -P 4 -I{} bash -c "{}"
