#!/bin/bash
set -e

# convergence and iterations for case 0,1,2 for unfem
# generate meshes/trapN.{is,vec} for N=1,...,10 first
# run as:
#   ./unfem-conv.sh &> unfem-conv.txt
# use PETSC_ARCH with --with-debugging=0 (for speed; time not measured)
# results:  hand edits give p4pdes-book/figs/unfem-conv.txt
# figure-generation:  p4pdes-book/figs/unfem-conv.py

function runcase() {
    CMD="../unfem -un_case $1 -un_mesh ../meshes/trap$2 $3"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "inear solve" tmp.txt
    grep "result" tmp.txt
}

# case 0 (linear); case 2 gives same iterations and errors
for LEV in 1 2 3 4 5 6 7 8 9 10; do
    runcase 0 $LEV "-snes_type ksponly -ksp_rtol 1.0e-10 -ksp_converged_reason"
done

# case 1 (nonlinear): use Picard iteration
for LEV in 1 2 3 4 5 6 7 8 9 10; do
    runcase 1 $LEV "-snes_rtol 1.0e-10 -ksp_rtol 1.0e-10 -snes_converged_reason -ksp_converged_reason"
done

