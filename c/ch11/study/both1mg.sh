#!/bin/bash
set -e

# number of iterations from use of classical GS-smoothed V-cycle multigrid on
# 1D advection-diffusion in more-or-less advection-dominated (eps=1/25,1/200) cases
# using first-order upwinding at all levels

# this script generated p4pdes-book/figs/both1mg.txt which is
# loaded by figure script p4pdes-book/figs/both1mg.py

COMMON="-snes_type ksponly -da_grid_x 5 -ksp_converged_reason -ksp_type richardson -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"
LEVELS="1 2 3 4 5 6 7 8 9 10 11 12"

function runcase() {
    CMD="../both1 $COMMON -b1_eps $1 -da_refine $2"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "done on" tmp.txt | awk '{print $3}'          # mx for grid
    grep "solve converged" tmp.txt | awk '{print $8}'  # KSP iterations
}

for LEV in $LEVELS; do
    runcase 0.04 "${LEV}"
done
echo
for LEV in $LEVELS; do
    runcase 0.005 "${LEV}"
done

