#!/bin/bash
set -e

# attempt to reproduce results on first table on page 27 of Smith et al 1996 "Domain Decomposition ..."

# note that numbers will be different because instead of "10^-2 reduction
# in error", probably in inf-norm (page 15), as in the table, here our criterion
# is 10^-2 reduction in preconditioned 2-norm (see man page for KSPSetNormType())

# do
#   ./asmaddmult.sh 4
# for N=33 (33x33 grid) and
#   ./asmaddmult.sh 5
# for N=65 (65x65 grid)

LEV=$1

COMMON="-fsh_dim 2 -ksp_type gmres -ksp_gmres_restart 10 -fsh_problem manuexp -snes_type ksponly -ksp_converged_reason -ksp_rtol 1.0e-2"

ASM="-pc_type asm -sub_pc_type lu -pc_asm_type basic"

#  -pc_asm_dm_subdomains: <FALSE> Use DMCreateDomainDecomposition() to define subdomains (PCASMSetDMSubdomains)
#  -pc_asm_blocks <-1>: Number of subdomains (PCASMSetTotalSubdomains)
#  -pc_asm_overlap <1>: Number of grid points overlap (PCASMSetOverlap)
#  -pc_asm_type <RESTRICT> Type of restriction/extension (choose one of) NONE RESTRICT INTERPOLATE BASIC (PCASMSetType)
#  -pc_asm_local_type <ADDITIVE> Type of local solver composition (choose one of) ADDITIVE MULTIPLICATIVE SYMMETRIC_MULTIPLICATIVE SPECIAL SCHUR (PCASMSetLocalType)

function runcase() {
  CMD="../fish $COMMON $1"
  #echo $CMD   # uncomment to show command, and add -snes_view to command to show solver
  rm -f tmp.txt
  $CMD &> tmp.txt
  grep iterations tmp.txt
  rm -f tmp.txt
}

for BL in 2 4 8; do
    for TYPE in multiplicative additive; do
        for OVER in 1 2 4; do
            echo "asm: $BL blocks, $TYPE, overlap $OVER"
            runcase "$ASM -da_refine $LEV -pc_asm_blocks $BL -pc_asm_local_type $TYPE -pc_asm_overlap $OVER"
        done
    done
done
echo "ilu:"
runcase "-da_refine $LEV -pc_type ilu"
echo "ssor:"
runcase "-da_refine $LEV -pc_type sor -pc_sor_symmetric"

