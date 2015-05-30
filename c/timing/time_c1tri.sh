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
  /usr/bin/time -f "%e" mpiexec -n $1 ../c1tri -tri_m 10000000 -ksp_type $2 -pc_type $3 &> tmp
  cat tmp |tail -n 1 &> c1tri/$2.$3.$1
  rm tmp
}

set -x

for N in 1 4; do
  run $N preonly lu
  for KSP in gmres cg; do
    for PC in ilu none jacobi; do
    FIXME
    done
  done
done

set +x
