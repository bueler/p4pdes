#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# weak scaling GMG smoothers for problem GLAZE using eps=1/100 and centered

COMMON="-bth_eps 0.01 -bth_problem glaze -bth_limiter centered -bth_none_on_peclet -snes_type ksponly -ksp_type bcgs -pc_type mg -ksp_converged_reason"
SMOOTH="-mg_levels_ksp_type richardson -mg_levels_pc_type asm -mg_levels_sub_pc_type sor"
COARSE="-da_grid_x 17 -da_grid_y 17"

function runcase() {
    CMD="mpiexec -n $1 ../both $COMMON $SMOOTH $COARSE -da_refine $2 -log_view"
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
LEV=4
for Z in 1 2 3 4; do
    runcase $P $LEV
    P=$(( $P * 4 ))
    LEV=$(( $LEV + 1))
done

