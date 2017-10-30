#!/bin/bash
set -e

# demo 2-level Dryja-Widlund type method, comparing to ASM+LU

# size of subdomain is fixed 65x65 by "mpiexec -n 4^(LEV-5)" for LEV = 5 6 7 8
# show that KSP iterations are h-independent

DW="-pc_type mg -pc_mg_levels 2 -pc_mg_type additive -mg_levels_ksp_type preonly -mg_levels_pc_type asm -mg_levels_sub_pc_type lu -mg_coarse_ksp_type preonly -mg_coarse_pc_type redundant -mg_coarse_redundant_pc_type lu"

ASMLU="-pc_type asm -sub_pc_type lu"

COMMON="-fsh_dim 2 -snes_type ksponly -ksp_type gmres -ksp_rtol 1.0e-10 -ksp_converged_reason -log_view"

function runcase() {
    CMD="mpiexec -n $1 ../fish $COMMON $2 -da_refine $3"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "problem manuexp" tmp.txt | awk '{print $4}'         # mx for grid
    grep "error |u-uexact|_inf" tmp.txt | awk '{print $4}' | sed 's/,*$//g' # numerical error; strip comma
    grep "Linear solve converged" tmp.txt | awk '{print $8}'  # KSP iterations
}

# NUMBER OF GRID POINTS IN SUBDOMAIN FIXED (thus H)
for PRECOND in "${DW}" "${ASMLU}"; do
    echo
    runcase  4 "${PRECOND}" 6
    echo
    runcase 16 "${PRECOND}" 7
    echo
    runcase 64 "${PRECOND}" 8
    #echo
    #runcase 512 9                 # mpiexec on WORKSTATION fails with 512 processes
                                   # this case can be tried in ch13/cluster.sh
done

