#!/bin/bash
set -e

# studies condition numbers from phelm.c for two coarse grids and 1.2 <= p <= 5.5

# run as:
#   ./phelmcond.sh &> phelmcond.txt
# use PETSC_ARCH with --with-debugging=0 (for convenience)

# results & figure-generation:  see p4pdes-book/figs/phelmcond.txt|py

EPS=1.0e-4
KSPRTOL=1.0e-12
GMRESRESTART=1000
SNESRTOL=1.0e-5

function runcase() {
    CMD="../phelm -snes_converged_reason -ksp_converged_reason -snes_rtol $SNESRTOL -ph_eps $EPS -pc_type none -ksp_rtol $KSPRTOL -ksp_gmres_restart $GMRESRESTART -ksp_compute_singularvalues -da_refine $1 -ph_p $2"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD -log_view &> tmp.txt
    grep -B 2 "Nonlinear solve" tmp.txt  # output of -{snes,ksp}_converged_reason and last cond number
    grep "numerical error" tmp.txt
}

for LEV in 3 5; do
    for PP in 1.2 1.4 1.6 1.8 2 2.5 3 3.5 4 4.5 5 5.5; do
        echo "p = $PP"
        runcase $LEV $PP
    done
    echo
done

