#!/bin/bash
set -e

# convergence and function evaluations for case 0 for unfem using -snes_fd|mf
# generate meshes/trapN.{is,vec} for N=1,...,9 first
# run as:
#   ./unfem-fdmf.sh &> unfem-fdmf.txt
# use PETSC_ARCH with --with-debugging=0 (for speed; time not measured)
# results:  hand edits give p4pdes-book/figs/unfem-fdmf.txt
# figure-generation:  p4pdes-book/figs/unfem-fdmf.py

function runcase() {
    CMD="../unfem $1 -un_mesh ../meshes/trap$2 -snes_converged_reason -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "inear solve" tmp.txt
    grep "result" tmp.txt
    grep "SNESFunctionEval" tmp.txt
}

for LEV in 1 2 3 4 5 6 7; do  # LEV=7 gives diverged
    runcase "-snes_fd" $LEV
done

for LEV in 1 2 3 4 5 6 7 8 9; do
    runcase "-snes_mf" $LEV
done


