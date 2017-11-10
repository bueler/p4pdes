#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 3D

# run as:
#   ./optimal3.sh &> optimal3.txt
# use PETSC_ARCH with --with-debugging=0

# limitations: going to -da_refine 8 gives out of memory on WORKSTATION with 16Gb

# results:  see p4pdes-book/figs/optimal3.txt
# figure-generation:  see p4pdes-book/figs/optimal.py

# choose (uncomment) *one* of the following:
ADD=
#ADD=-pc_mg_galerkin
#ADD=-snes_fd_color      # FIXME makes more sense to show for minimal
#ADD=-snes_mf_operator   # ditto

for LEV in 3 4 5 6 7; do
    CMD="../fish -fsh_dim 3 -snes_type ksponly -ksp_type cg -pc_type mg $ADD -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

