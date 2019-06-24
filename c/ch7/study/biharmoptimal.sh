#!/bin/bash
set -e

# demonstrates optimality of a coupled-form biharmonic equation solver using
# GMRES and GMG either with multiplicative/additive fieldsplit or monolithically

# run as:
#   ./biharmoptimal.sh &> biharmoptimal.txt
# use PETSC_ARCH with --with-debugging=0  (but only flops are measured, not time)

# results & figure-generation:  see p4pdes-book/figs/biharmoptimal.txt|py

function runcase() {
    CMD="../biharm -da_refine $1 $2 -ksp_type gmres -ksp_converged_reason -log_view"
    #echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -A 1 "done on" tmp.txt
    grep "solve converged" tmp.txt
    grep "Flop:  " tmp.txt
}

LEVLIST="4 5 6 7 8 9 10"  # 33x33 to 2049x2049

FSGMG="-pc_type fieldsplit -fieldsplit_v_pc_type mg -fieldsplit_u_pc_type mg -fieldsplit_v_pc_mg_galerkin -fieldsplit_u_pc_mg_galerkin"

echo "===== monolithic ====="
for LEV in $LEVLIST; do
    PC="-pc_type mg -pc_mg_levels $LEV -pc_mg_galerkin"
    runcase $LEV "$PC"
done

echo
echo "===== multiplicative fieldsplit ====="
for LEV in $LEVLIST; do
    PC="$FSGMG -fieldsplit_v_pc_mg_levels $LEV -fieldsplit_u_pc_mg_levels $LEV"
    runcase $LEV "$PC"
done

echo
echo "===== additive fieldsplit ====="
for LEV in $LEVLIST; do
    PC="$FSGMG -pc_fieldsplit_type additive -fieldsplit_v_pc_mg_levels $LEV -fieldsplit_u_pc_mg_levels $LEV"
    runcase $LEV "$PC"
done

