#!/bin/bash

set +x

LSEXEC=$1
EXECNAME=$2
MPI="mpiexec"

function run() {
  rm -f tmp
  set -x
  $MPI -n $1 $LSEXEC -ksp_rtol 1.0e-10 -ksp_converged_reason -log_view -ksp_type $2 -pc_type $3 $4 &> tmp
  set +x
  grep "Linear solve " tmp
  grep "Time (sec):" tmp | awk '{print $3}'
  rm -f tmp
}

for PC in lu cholesky; do
    run 1 preonly $PC ""
done
for N in 1 4; do
    run $N richardson jacobi ""
done
for PC in none jacobi; do
    for N in 1 4; do
        run $N gmres $PC ""
    done
done
run 1 gmres ilu ""
run 4 gmres bjacobi "-sub_pc_type ilu"
for PC in none jacobi; do
    for N in 1 4; do
        run $N cg $PC ""
    done
done
run 1 cg icc ""
run 4 cg bjacobi "-sub_pc_type icc"

