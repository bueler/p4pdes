#!/bin/bash
set -e

# compares objective-only runs of p=2 case of phelm.c

# run as:
#   ./noresidaul.sh &> noresidual.txt
# use PETSC_ARCH with --with-debugging=0 (for convenience)

# results & figure-generation:  see p4pdes-book/figs/noresidual.txt|py

RTOL=1.0e-5

function runcase() {
    CMD="../phelm -snes_converged_reason -snes_rtol $RTOL -ph_no_residual -snes_fd_function -da_refine $1 $2"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "Nonlinear solve" tmp.txt
    grep "numerical error" tmp.txt
}

# for each method go to first failure

for METHOD in "-snes_fd_color"; do
    for LEV in 0 1 2 3 4 5; do
        runcase $LEV "$METHOD"
    done
done

for METHOD in "-snes_fd_color -snes_mf_operator"; do
    for LEV in 0 1 2 3 4; do
        runcase $LEV "$METHOD"
    done
done

for METHOD in "-snes_type qn" "-snes_type ncg"; do
    for LEV in 0 1 2 3 4 5 6; do
        runcase $LEV "$METHOD"
    done
done



