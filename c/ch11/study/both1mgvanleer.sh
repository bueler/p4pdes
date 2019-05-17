#!/bin/bash
set -e

# includes timing; run --with-debugging=0

# number of iterations and timing to solve 1D advection-diffusion equation
# for eps=1/200; comparison of van Leer solves using -snes_fd_color
# and -snes_mf_operator; linear solver is GMRES plus these PCs
#    * LU solver (optimal)
#    * classical GS-smoothed V-cycle multigrid, but with first-order upwind
#      (re)discretization except on finest grid

COMMON="-b1_eps 0.005 -b1_limiter vanleer -snes_fd_color -snes_mf_operator -snes_converged_reason"
MG="-b1_none_on_down -ksp_type gmres -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"
LU="-ksp_type gmres -pc_type lu"
LEVELS="10 11 12 13 14 15 16 17 18 19 20"

function runcase() {
    CMD="../both1 $COMMON $1 -da_refine $2 -log_view"
    #echo $CMD
    rm -rf tmp.txt
    $CMD &> tmp.txt
    #cat tmp.txt
    #grep "solve converged" tmp.txt
    grep "done on" tmp.txt | awk '{print $3}'          # mx for grid
    grep "solve converged" tmp.txt | awk '{print $8}'  # KSP iterations
    grep "SNESSolve" tmp.txt | awk '{print $4}'        # time
}

for LEV in $LEVELS; do
    runcase "${LU} " "${LEV}"
done
for LEV in $LEVELS; do
    runcase "${MG} " "${LEV}"
done

