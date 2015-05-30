#!/bin/bash

set +x

function run() {
  rm -f tmp
  set -x
  /usr/bin/time -f "%e" mpiexec -n $1 ../../c1tri -tri_m 10000000 -ksp_type $2 -pc_type $3 $4 &> tmp
  set +x
  NAME=$2.$3.$1
  rm -f $NAME
  cat tmp |tail -n 1 &> $NAME
  rm tmp
  cat $NAME
}

run 1 preonly lu ""
for PC in none ilu jacobi; do
    run 1 gmres $PC ""
done
for PC in none ilu jacobi icc; do
    run 1 cg $PC ""
done
run 4 gmres none ""
run 4 gmres jacobi ""
run 4 gmres bjacobi "-sub_pc_type ilu"
run 4 cg none ""
run 4 cg jacobi ""
run 4 cg bjacobi "-sub_pc_type ilu"
run 4 cg bjacobi "-sub_pc_type icc"

