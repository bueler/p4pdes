#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 2D

# run as:
#   ./optimal2.sh &> optimal2.txt
# use PETSC_ARCH with --with-debugging=0

# limitation: going to -da_refine 12 gives out of memory on WORKSTATION with 16Gb

# results:  see p4pdes-book/figs/optimal2.txt
# figure-generation:  see p4pdes-book/figs/optimal.py

for LEV in 5 6 7 8 9 10 11; do
    CMD="../fish -fsh_dim 2 -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

