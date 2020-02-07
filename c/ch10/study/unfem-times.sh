#!/bin/bash
set -e

# solver time for case 0 of unfem, either CG+ICC or CG+GAMG
#
# run as:
#   cd c/ch10/
#   make unfem                        # use PETSC_ARCH with --with-debugging=0
#   ./refinetraps.sh meshes/trap 12   # generate meshes/trapN.{is,vec} for N=1,...,12
#   cd study/
#   ./unfem-times.sh &> unfem-times.txt
#
# results:  hand edits give p4pdes-book/figs/unfem-times.txt
# figure-generation:  p4pdes-book/figs/unfem-times.py

function run() {
    CMD="../unfem -un_case 0 -un_mesh ../meshes/trap$1 $2 -snes_type ksponly -ksp_rtol 1.0e-10 -ksp_converged_reason -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "Linear solve" tmp.txt
    grep "result" tmp.txt
    grep "Flop:           " tmp.txt
    grep "Time (sec):     " tmp.txt
    # read time percentages from these lines
    grep "Read mesh      :" tmp.txt
    grep "Set-up         :" tmp.txt
    grep "Solver         :" tmp.txt
}

# case 0 (linear, homo neumann) with CG+ICC
for LEV in 1 2 3 4 5 6 7 8 9 10 11; do
    run $LEV "-pc_type icc"
done

# case 0 (linear, homo neumann) with CG+GAMG
for LEV in 1 2 3 4 5 6 7 8 9 10 11 12; do
    run $LEV "-pc_type gamg"
done

