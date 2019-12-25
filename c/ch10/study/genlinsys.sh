#!/bin/bash

# generate N= linear system for chapter 2 timing results
# run in ch10/:
#   ./refinetraps.sh meshes/trap 8
# then run this script here in ch10/study/ once:
#   ./genlinsys.sh ../meshes/trap8

../unfem -un_mesh $1 -ksp_type cg -pc_type gamg -ksp_converged_reason -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat

