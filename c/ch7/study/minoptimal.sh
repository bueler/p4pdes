#!/bin/bash
set -e

# demonstrates optimality of minimal surface equation Newton-Krylov solver
# with grid sequencing, -snes_fd_color, GMRES, and GMG preconditioning

# main concern is issue iii), the smoothness of solutions, so we vary H in tent
# and c in catenoid

# run as:
#   ./minoptimal.sh &> minoptimal.txt
# use PETSC_ARCH with --with-debugging=0   (but only flops are measured, not time)

# results & figure-generation:  see p4pdes-book/figs/minoptimal.txt|py

function runcase() {
    CMD="../minimal -snes_fd_color -pc_type mg -snes_converged_reason -log_view -snes_grid_sequence $1 -mse_problem $2 $3"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -B 1 "grid and problem" tmp.txt
    grep "Flop:  " tmp.txt
}

for HH in 0.1 1 10; do
    for LEV in 5 6 7 8 9; do
        runcase $LEV tent "-mse_tent_H $HH"
    done
done

for CC in 1.1 1.01 1.0001; do
    for LEV in 5 6 7 8 9; do
        runcase $LEV catenoid "-mse_catenoid_c $CC"
    done
done

