#!/bin/bash

# generates data for a table in chapter 3
# run as
#    $ export PETSC_ARCH=linux-c-opt
#    $ make poisson
#    $ cd study/
#    $ ./time.sh

set +x

REFINE=5

function run() {
  /usr/bin/time -f "%e" ../poisson -da_refine $REFINE -ksp_converged_reason -ksp_type $1 -pc_type $2 $3
}

run gmres none
run gmres ilu
run gmres ilu "-ksp_gmres_restart 200"
run cg none
run cg jacobi
run cg icc
run cg icc "-ksp_rtol 1.0e-14"
run preonly cholesky
run minres none

