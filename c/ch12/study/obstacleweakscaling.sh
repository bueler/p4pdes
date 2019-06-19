#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# with hand-editing, generates p4pdes-book/figs/obstacleweakscaling.txt
# see last table and figure in Chapter 12

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
    grep "solve converged " tmp.txt
    grep "Flop:  " tmp.txt
}

P=1
LEV=4
for Z in 1 2 3 4; do
    runcase $P $LEV
    P=$(( $P * 4 ))
    LEV=$(( $LEV + 1))
done

