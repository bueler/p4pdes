#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 2D

# run as:
#   ./optimal2.sh &> optimal2.txt
# use PETSC_ARCH with --with-debugging=0

# limitation: going to -da_refine 12 gives out of memory on WORKSTATION with 16Gb

# results & figure-generation:  see p4pdes-book/figs/optimal2.txt|.py

MINLEV=5
MAXLEV=${1:-11}  # expands to $1 if set, otherwise is 11

for (( LEV=MINLEV; LEV<=$MAXLEV; LEV++ )); do
    CMD="../fish -fsh_dim 2 -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

