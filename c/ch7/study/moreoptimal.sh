#!/bin/bash
set -e

# considers optimality of a CG+MG solver for 3D Poisson equation using
#     * V, W, full(V) cycles
#     * -pc_mg_galerkin   (V cycles only)

# use PETSC_ARCH with --with-debugging=0

# results and figure-generation:  see p4pdes-book/figs/moreoptimal.txt|py

MINLEV=3
MAXLEV=${1:-7}  # expands to $1 if set, otherwise is 7

for ADD in "" "-pc_mg_cycle_type w" "-pc_mg_type full" "-pc_mg_galerkin"; do
    echo "**** CASE: $ADD ****"
    for (( LEV=MINLEV; LEV<=$MAXLEV; LEV++ )); do
        CMD="../../ch6/fish -fsh_dim 3 -pc_type mg $ADD -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
        echo "COMMAND:  $CMD"
        rm -rf tmp.txt
        $CMD &> tmp.txt
        grep -C 1 "problem manuexp" tmp.txt
        grep SNESSolve tmp.txt
    done
    echo
done

