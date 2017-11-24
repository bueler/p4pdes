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

MINLEV=2
MAXLEV=${1:-6}  # expands to $1 if set, otherwise is 6

for PAR in "" "mpiexec -n 8" "mpiexec -n 64"; do
    echo "**** CASE: $PAR ****"
    for (( LEV=MINLEV; LEV<=$MAXLEV; LEV++ )); do
        CMD="$PAR ../fish -fsh_dim 3 $COARSE -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
        echo "COMMAND:  $CMD"
        rm -rf tmp.txt
        $CMD &> tmp.txt
        grep -C 1 "problem manuexp" tmp.txt
        grep "Flop:  " tmp.txt
    done
    echo
done

