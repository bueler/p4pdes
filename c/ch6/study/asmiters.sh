#!/bin/bash
set -e

# test ASM iterations with 2,3 dimensions and lu,ilu solution on blocks, with
#   NN   = 2,4,8    subdomains/processes
#   LEV  = 4,6,8    refinement (m=33,129,513) in 2D;  only m=33,129 in 3D
#   OVER = 0,1,4    overlap

COMMON="-snes_type ksponly -ksp_converged_reason -pc_type asm"

# notes:
#   * defaults to -ksp_type gmres -ksp_gmres_restart 30 -ksp_rtol 1.0e-5
#   * run with --with-debugging=0 configuration

#FIXME exercise: set GMRES300 and see reduced iters

#FIXME exercise: set CG+cholesky|icc and -ksp_max_it 1000 and see some DIVERGED_ITS

function runcase() {
  CMD="mpiexec -n $1 ../fish -fsh_dim $2 $COMMON -da_refine $3 -pc_asm_overlap $4 -sub_pc_type $5"
  echo $CMD   # uncomment to show command, and add -snes_view to command to show solver
  rm -f tmp.txt
  $CMD &> tmp.txt
  grep iterations tmp.txt
  rm -f tmp.txt
}

for NN in 2 4 8; do
    for OVER in 0 1 4; do
        for LEV in 4 6 8; do
            runcase $NN 2 $LEV $OVER lu
            runcase $NN 2 $LEV $OVER ilu
        done
    done
done

for NN in 2 4 8; do
    for OVER in 0 1 4; do
        for LEV in 4 6; do
            runcase $NN 3 $LEV $OVER ilu
        done
    done
done


