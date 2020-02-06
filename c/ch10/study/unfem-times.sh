#!/bin/bash
set -e

# solver time for case 0 of unfem, either CG+ICC or CG+GAMG
# generate meshes/trapN.{is,vec} for N=1,...,10 first
# run as:
#   ./unfem-times.sh &> unfem-times.txt
# use PETSC_ARCH with --with-debugging=0
# results:  hand edits give p4pdes-book/figs/unfem-times.txt
# figure-generation:  p4pdes-book/figs/unfem-times.py

function run() {
    CMD="../unfem -un_case $1 -un_mesh ../meshes/trap$2 $3 -snes_converged_reason -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "inear solve" tmp.txt
    grep "result" tmp.txt
    grep "Flop:           " tmp.txt
    grep "Time (sec):     " tmp.txt
    # read time percentages from these lines, especially "Solver"
    grep "     Main Stage:" tmp.txt
    grep "Read mesh      :" tmp.txt
    grep "Set-up         :" tmp.txt
    grep "Solver         :" tmp.txt
    grep "Residual eval  :" tmp.txt
    grep "Jacobian eval  :" tmp.txt
}

# case 0 (linear, homo neumann) with CG+ICC
for LEV in 1 2 3 4 5 6 7 8 9 10; do
    run 0 $LEV " "
done

# case 0 (linear, homo neumann) with CG+GAMG
for LEV in 1 2 3 4 5 6 7 8 9 10; do
    run 0 $LEV "-pc_type gamg"
done

