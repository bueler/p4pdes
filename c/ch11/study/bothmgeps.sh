#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# number of iterations and run time from BCGS with GS-smoothed V-cycle
# multigrid on 2D advection-diffusion, and first-order upwinding at all levels,
# in eps=1/10,1/200 cases

# WHY BCGS:
#for KSP in gmres cgs bcgs tfqmr; do timer ./both -bth_eps 0.005 -snes_type ksponly -pc_type mg -mg_levels_ksp_type richardson     -mg_levels_pc_type sor -mg_levels_pc_sor_forward -da_refine 10 -ksp_converged_reason -ksp_type $KSP; done
#  Linear solve converged due to CONVERGED_RTOL iterations 6
#done on 2049 x 2049 grid (problem = layer, eps = 0.005, limiter = none)
#numerical error:  |u-uexact|_2 = 2.6338e-03
#real 18.37
#  Linear solve converged due to CONVERGED_RTOL iterations 3
#done on 2049 x 2049 grid (problem = layer, eps = 0.005, limiter = none)
#numerical error:  |u-uexact|_2 = 2.6332e-03
#real 18.13
#  Linear solve converged due to CONVERGED_RTOL iterations 3
#done on 2049 x 2049 grid (problem = layer, eps = 0.005, limiter = none)
#numerical error:  |u-uexact|_2 = 2.6314e-03
#real 17.91
#  Linear solve converged due to CONVERGED_RTOL iterations 4
#done on 2049 x 2049 grid (problem = layer, eps = 0.005, limiter = none)
#numerical error:  |u-uexact|_2 = 2.6335e-03
#real 19.19

# no obvious advantage to adding -ksp_pc_side right

# this script generated p4pdes-book/figs/bothmgeps.txt which is
# loaded by figure script p4pdes-book/figs/bothmgeps.py

COMMON="-snes_type ksponly -ksp_type bcgs -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"
LEVELS="3 4 5 6 7 8 9 10 11"

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

