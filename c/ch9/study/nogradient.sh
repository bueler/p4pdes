#!/bin/bash
set -e

# compares objective-only runs of p=2 case of phelm.c

# run as:
#   ./nogradient.sh &> nogradient.txt
# use PETSC_ARCH with --with-debugging=0 (for convenience)

# results:  see p4pdes-book/figs/phelmnogradient.txt

RTOL=1.0e-6
MAXIT=200

function runcase() {
    CMD="../phelm -snes_converged_reason -snes_rtol $RTOL -snes_max_it $MAXIT -ph_no_gradient -snes_fd_function -da_refine $1 $2"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "Nonlinear solve" tmp.txt
    grep "done on" tmp.txt
    grep "numerical error" tmp.txt
}

# for each method go to 33x33 grid
# LEV:  0=2x2, 1=3x3, 2=5x5, 3=9x9, 4=17x17, 5=33x33, 6=65x65

for METHOD in "-snes_fd_color" "-snes_fd_color -snes_mf_operator" "-snes_type qn" "-snes_type ncg"; do
    for LEV in 0 1 2 3 4 5; do
        runcase $LEV "$METHOD"
    done
done

