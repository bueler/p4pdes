#!/bin/bash
set -e

# test ASM iterations with 2,3 dimensions and
#   LEV  = 4,5,6,7      refinement (m=33,65,129,257) in 2D;  only m=33,65 in 3D
#   OVER = 0,1,4,16     overlap
# for all runs uses 4 blocks and lu on subdomains

COMMON="-snes_type ksponly -ksp_converged_reason -pc_type asm -sub_pc_type lu"

# or exploit symmetry by using CG and CHOLESKY; recall need to turn on nested dissection:
#COMMON="-snes_type ksponly -ksp_converged_reason -ksp_type cg -pc_type asm -sub_pc_type cholesky -sub_pc_factor_mat_ordering_type nd"
# conclusion: a few more iterations, but a little bit faster, and all the same patterns

# notes:
#   * defaults to -ksp_type gmres -ksp_gmres_restart 30 -ksp_rtol 1.0e-5
#   * run with --with-debugging=0 configuration

function runcase() {
  CMD="mpiexec -n 4 ../fish $COMMON -fsh_dim $1 -da_refine $2 -pc_asm_overlap $3"
  echo $CMD   # uncomment to show command, and add -snes_view to command to show solver
  rm -f tmp.txt
  /usr/bin/time -f "real %e" $CMD &> tmp.txt
  grep iterations tmp.txt
  grep real tmp.txt
  rm -f tmp.txt
}

for OVER in 0 1 2 4; do
    for LEV in 4 5 6 7; do
        runcase 2 $LEV $OVER
    done
    for LEV in 4 5; do
        runcase 3 $LEV $OVER
    done
done

