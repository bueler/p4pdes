#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 3D

# run as:
#   ./optimal3.sh &> optimal3.txt
# use PETSC_ARCH with --with-debugging=0

# limitations: going to -da_refine 8 gives out of memory on WORKSTATION with 16Gb

# results & figure-generation:  see p4pdes-book/figs/optimal3.txt|.py

MINLEV=3
MAXLEV=${1:-7}  # expands to $1 if set, otherwise is 7

for (( LEV=MINLEV; LEV<=$MAXLEV; LEV++ )); do
    CMD="../fish -fsh_dim 3 -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

