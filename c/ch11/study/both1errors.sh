#!/bin/bash
set -e

# L2 errors of 1D advection-diffusion in advection-dominated (eps=0.01) case
# shows first-order upwind is eventually O(h^1), centered is O(h^2),
# vanleer is eventually O(h^2), and vanleer combines best of others

COMMON="-snes_rtol 1.0e-11 -snes_converged_reason -ksp_type preonly -pc_type lu"
LEVELS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"

function runcase() {
    CMD="../both1 $COMMON -b1_limiter $1 -b1_jac_limiter $2 -da_refine $3"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "done on" tmp.txt | awk '{print $3}'          # mx for grid
    grep "solve converged" tmp.txt | awk '{print $8}'  # KSP iterations
    grep "|u-uexact|_2" tmp.txt | awk '{print $8}'     # numerical error
}

for LEV in $LEVELS; do
    echo
    runcase none none "${LEV}"
done
for LEV in $LEVELS; do
    echo
    runcase centered centered "${LEV}"
done
for LEV in $LEVELS; do
    echo
    runcase vanleer none "${LEV}"
done

