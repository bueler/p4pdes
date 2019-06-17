#!/bin/bash
set -e

# generates p4pdes-book/figs/obstacleweakscaling.txt; see last table and figure in Chapter 12

# FIXME run with --with-debugging=1 configuration, and lower resolution than desired, because of PETSc issue #306
# see PR #1785 which seems to resolve it

# weak scaling for obstacle:  grid-sequencing + RS + GMG + ASM/SSOR

COMMON="-snes_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type asm -mg_levels_sub_pc_type sor"
COARSE="-da_grid_x 17 -da_grid_y 17"

function runcase() {
    CMD="mpiexec -n $1 ../obstacle $COMMON $SMOOTH $COARSE -snes_grid_sequence $2 -log_view"
    echo $CMD
    rm -rf tmp.txt
    $CMD &> tmp.txt
    #cat tmp.txt
    grep "done on " tmp.txt
    #grep "done on " tmp.txt | awk '{print $3}'          # mx for grid
    grep "solve converged " tmp.txt
    #grep "solve converged " tmp.txt | awk '{print $8}'  # KSP iterations
    grep "Flop:  " tmp.txt
}

P=1
LEV=3  # FIXME once #306 is fixed, rerun with --with-debugging=0 and LEV=4
for Z in 1 2 3 4; do
    runcase $P $LEV
    P=$(( $P * 4 ))
    LEV=$(( $LEV + 1))
done

