#!/bin/bash
set -e

# demonstrates optimality of minimal surface equation Newton-Krylov solver
# with grid sequencing and GMG preconditioning

# run as:
#   ./minoptimal.sh &> minoptimal.txt
# use PETSC_ARCH with --with-debugging=0

# limitation: FIXME gives out of memory on WORKSTATION with 16Gb

# results:  see p4pdes-book/figs/FIXME
# figure-generation:  see p4pdes-book/figs/FIXME

# choose (uncomment) *one* of the following:
PROB=tent               # note low regularity at boundary,
                        # so linear problems get worse as grid is refined
#PROB=catenoid

# choose (uncomment) *one* of the following:
JAC=-snes_fd_color
#JAC=-snes_mf_operator

# you can also add -pc_mg_galerkin

for LEV in 5 6 7 8 9; do
#for LEV in 5 6 7 8 9 10; do    # on finest level, FD jacobian seems to lose quality
    CMD="../minimal -mse_problem $PROB $JAC -ksp_type cg -pc_type mg -snes_converged_reason -ksp_converged_reason -snes_grid_sequence $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -B 2 "done on" tmp.txt
    grep SNESSolve tmp.txt
done

