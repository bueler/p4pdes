#!/bin/bash
set -e

# solver iterations and flops for case 1 of unfem using CG+AMG and one of
# three nonlinear strategies:
#   Picard (analytical matrix) iteration
#   -snes_fd_color
#   -snes_mf_operator with Picard as preconditioner material
# (note: individual runs will show the very different residual norm histories)

# run as:
#   cd c/ch10/
#   make unfem                        # use PETSC_ARCH with --with-debugging=0
#   ./refinetraps.sh meshes/trap 12   # generate meshes/trapN.{is,vec} for N=1,...,12
#   cd study/
#   ./unfem-nonlin.sh &> unfem-nonline.txt
# results:  hand edits give p4pdes-book/figs/unfem-nonlin.txt
# figure-generation:  p4pdes-book/figs/unfem-nonlin.py

function run() {
    CMD="../unfem -un_case 1 -un_mesh ../meshes/trap$1 $2 -pc_type gamg -snes_rtol 1.0e-8 -ksp_rtol 1.0e-8 -snes_converged_reason -ksp_converged_reason -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "Linear solve" tmp.txt
    grep "Nonlinear solve" tmp.txt
    grep "case 1 result" tmp.txt
    grep "Flop:           " tmp.txt
    grep "Time (sec):     " tmp.txt
}

echo "********** Picard ***********"
for LEV in 3 4 5 6 7 8 9 10 11; do
    run $LEV ""
done
echo

echo "********** Newton by _fd_color ***********"
for LEV in 3 4 5 6 7 8 9 10 11; do
    run $LEV "-snes_fd_color"
done
echo

echo "********** JFNK with Picard matrix ***********"
for LEV in 3 4 5 6 7 8 9 10; do
    run $LEV "-snes_mf_operator"
done
echo


