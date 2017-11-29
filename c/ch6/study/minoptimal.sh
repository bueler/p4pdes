#!/bin/bash
set -e

# demonstrates optimality of minimal surface equation Newton-Krylov solver
# with grid sequencing and CG and GMG preconditioning

# run as:
#   ./minoptimal.sh &> minoptimal.txt
# use PETSC_ARCH with --with-debugging=0

# results & figure-generation:  see p4pdes-book/figs/minoptimal.txt|py

# choose (uncomment) *one* of the following:
PROB=catenoid
#PROB=tent     # low regularity at boundary so linear problems get worse as grid is refined

# choose (uncomment) *one* of the following:
JAC=-snes_mf_operator
#JAC=-snes_fd_color   # faster, but on finest levels FD jacobian loses quality

for LEV in 5 6 7 8 9 10; do
    CMD="../minimal -mse_problem $PROB $JAC -ksp_type cg -pc_type mg -snes_converged_reason -ksp_converged_reason -snes_grid_sequence $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -B 2 "done on" tmp.txt
    grep SNESSolve tmp.txt
done

