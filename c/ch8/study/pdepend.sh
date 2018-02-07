#!/bin/bash
set -e

# studies convergence and solver complexity of phelm.c for various p values
# serial

# run as:
#   ./pdepend.sh &> pdepend.txt
# use PETSC_ARCH with --with-debugging=0 (for convenience)

# results & figure-generation:  see p4pdes-book/figs/pdepend.txt|py

PRANGE="1.4 1.8 2.5 4 8"

LEVRANGE="4 5 6 7 8 9 10"
# 3 = 9x9
# 4 = 17x17
# 5 = 33x33
# 6 = 65x65
# 7 = 129x129
# 8 = 257x257
# 9 = 513x513
#10 = 1025x1025
#11 = 2049x2049

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
    grep "Flop:  " tmp.txt          # total number of flops
    grep "Time (sec):" tmp.txt
}

for PP in $PRANGE; do
    echo "======================================================"
    echo "p = $PP"
    for LEV in $LEVRANGE; do
        echo
        runcase $LEV $PP
    done
    echo
done

