#!/bin/bash
set -e

# studies phelm.c for various p values

# run as:
#   ./pdependence.sh &> pdependence.txt
# use PETSC_ARCH with --with-debugging=0 (for convenience)

# results & figure-generation:  see p4pdes-book/figs/FIXME

RTOL=1.0e-5
EPS=1.0e-4

function runcase() {
    CMD="../phelm -snes_converged_reason -snes_rtol $RTOL -ksp_type cg -pc_type mg -snes_grid_sequence $1 -ph_p $2 -ph_eps $EPS"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD -log_view &> tmp.txt
    grep "Nonlinear solve" tmp.txt  # allows count of Newton iters and CONVERGED/DIVERGED
    grep "numerical error" tmp.txt
    grep "PCApply" tmp.txt          # total number of linear iterations (sort of)
}

for PP in 1.1 1.5 2 4 10; do
    echo "p = $PP"
    for LEV in 3 4 5 6 7 8; do
        echo
        runcase $LEV $PP
    done
    echo
done

