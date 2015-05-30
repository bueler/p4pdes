#!/bin/bash

#preonly & lu     &  \\
#gmres   & ilu    &  \\
#        & none   & \\
#        & jacobi & \\
#cg      & ilu    &  \\
#        & none   & \\
#        & jacobi & \\
#        & icc    &

function run() {
  rm -f tmp
  /usr/bin/time -f "%e" mpiexec -n $1 ../c1tri -tri_m 10000000 $2 &> tmp
  cat tmp |tail -n 1 &> $3
}

set -x

for N in 1 4; do
  for PC in ilu none jacobi; do
  rm -f tmp
  run $N '-ksp_type preonly -pc_type none' c1tri/preonly.lu.$N
  done
done

set +x
