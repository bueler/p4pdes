#!/bin/bash

set +x

EXEC=$1

function run() {
  rm -f tmp
  set -x
  /usr/bin/time -f "%e" mpiexec -n $1 $EXEC -tri_m 10000000 -ksp_rtol 1.0e-10 -ksp_converged_reason -ksp_type $2 -pc_type $3 $4 &> tmp
  set +x
  cat tmp
  NAME=$2.$3.$1
  rm -f $NAME
  cat tmp |tail -n 1 &> $NAME
  rm tmp
}

for PC in lu cholesky; do
    run 1 preonly $PC ""
done
for N in 1 4; do
    run $N richardson jacobi ""
done
for KSP in gmres cg; do
    for N in 1 4; do
        for PC in none jacobi; do
            run $N $KSP $PC ""
        done
    done
done
run 1 gmres ilu ""
run 4 gmres bjacobi "-sub_pc_type ilu"
run 1 cg icc ""
run 4 cg bjacobi "-sub_pc_type icc"

