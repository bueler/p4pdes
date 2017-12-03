#!/bin/bash
set -e

# on 1,8,64 processors, consider optimality (increasing N values) of a
# CG+MG solver (V cycles) for the 3D Poisson equation

# use PETSC_ARCH with --with-debugging=0

# results and figure-generation:  see p4pdes-book/figs/paroptimal.txt|.py

# the 3x3x3 coarse grid used in optimal3.sh and moreoptimal.sh cannot be
# divided among 64 processes
# with a 5x5x5 grid, levels 2--6 are the same fine grids as levels 3--7 in
# those other scripts, but the number of levels in a V cycle is one less here
COARSE="-da_grid_x 5 -da_grid_y 5 -da_grid_z 5"

function runcase() {
    CMD="$1 ../fish -fsh_dim 3 -snes_type ksponly -ksp_type cg -ksp_rtol 1.0e-10 -ksp_converged_reason $COARSE -da_refine $2 $3 -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep "Flop:  " tmp.txt
}

MINLEV=2
MAXLEV=${1:-6}  # expands to $1 if set, otherwise is 6

for PAR in "" "mpiexec -n 8" "mpiexec -n 64"; do
    echo "**** CASE: $PAR ****"
    SOLVE="-pc_type mg"
    for (( LEV=MINLEV; LEV<=$MAXLEV; LEV++ )); do
        runcase "${PAR}" $LEV "${SOLVE}"
    done
    echo
    SOLVE="-pc_type asm -sub_pc_type icc"
    for (( LEV=MINLEV; LEV<=5; LEV++ )); do
        runcase "${PAR}" $LEV "${SOLVE}"
    done
    echo
done

