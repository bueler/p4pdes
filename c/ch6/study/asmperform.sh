#!/bin/bash
set -e

# measure flops in ASM iterations with 3 dimensions and 8 processes
# compare Cholesky and ICC solves on the subgrids
# convenience:  run with --with-debugging=0 configuration

COMMON="-fsh_dim 3 -ksp_converged_reason -ksp_type cg -pc_type asm -log_view"
CHOL="-sub_pc_type cholesky -sub_pc_factor_mat_ordering_type nd"
ICC="-sub_pc_type icc"

function runcase() {
    CMD="mpiexec -n 8 ../fish $COMMON $1 -da_refine $2 -pc_asm_overlap $3"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "problem manuexp" tmp.txt
    grep iterations tmp.txt
    grep KSPSolve tmp.txt
}

runcase "$CHOL" 4 1
runcase "$CHOL" 5 2
runcase "$CHOL" 6 4

runcase "$ICC" 4 1
runcase "$ICC" 5 2
runcase "$ICC" 6 4
runcase "$ICC" 7 8

