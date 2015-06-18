#!/bin/bash

set +x

function run() {
  rm -f tmp
  set -x
  /usr/bin/time -f "%e" ../../poisson -da_refine 5 -ksp_converged_reason -ksp_type $1 -pc_type $2 $3 &> tmp
  set +x
  cat tmp
  NAME=$1.$2$4
  rm -f $NAME
  cat tmp | tail -n 1 &> $NAME
  echo "&" >> $NAME
  cat tmp | grep iterations | sed "s/.*iterations //" | sed "s/ .*//" >> $NAME
  rm tmp
}

run gmres ilu
run gmres none
run gmres ilu "-ksp_gmres_restart 200" .restart
run cg none
run cg jacobi
run cg icc
run minres none
run preonly cholesky
run cg icc "-ksp_rtol 1.0e-14" .tight

