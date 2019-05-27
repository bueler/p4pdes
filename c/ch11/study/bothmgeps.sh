#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# number of iterations and run time from classical GS-smoothed V-cycle
# multigrid on 2D advection-diffusion in eps=1/10,1/200 cases

# first-order upwinding at all levels

# this script generated p4pdes-book/figs/bothmgeps.txt which is
# loaded by figure script p4pdes-book/figs/bothmgeps.py

COMMON="-snes_type ksponly -ksp_converged_reason -ksp_type richardson -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"
LEVELS="4 5 6 7 8 9 10 11"

function runcase() {
    CMD="../both $COMMON -bth_eps $1 -da_refine $2"
    rm -rf tmp.txt
    /usr/bin/time -f "real %e" $CMD &> tmp.txt
    #cat tmp.txt
    grep "done on " tmp.txt | awk '{print $3}'          # mx for grid
    grep "solve converged " tmp.txt | awk '{print $8}'  # KSP iterations
    grep "real " tmp.txt | awk '{print $2}'             # run time
}

for LEV in $LEVELS; do
    runcase 0.1 "${LEV}"
done
echo
for LEV in $LEVELS; do
    runcase 0.005 "${LEV}"
done

