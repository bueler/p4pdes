#!/bin/bash
set -e

# residuals from Gauss-Seidel sweeps for different eps and with forward/backward/ssor

# this script generates both1gs_{eps10,eps200,forward,backward,ssor}.txt which are
# loaded by figure script p4pdes-book/figs/both1gs.py

COMMON="-da_refine 4 -b1_limiter none -b1_jac_limiter none -snes_type ksponly -ksp_type richardson -pc_type sor -ksp_rtol 1.0e-6"

OUT=both1gs_eps10.txt
echo "generating $OUT ..."
../both1 $COMMON -b1_eps 0.1 -pc_sor_forward -ksp_monitor_true_residual &> $OUT
OUT=both1gs_eps200.txt
echo "generating $OUT ..."
../both1 $COMMON -b1_eps 0.005 -pc_sor_forward -ksp_monitor_true_residual &> $OUT

OUT=both1gs_forward.txt
echo "generating $OUT ..."
../both1 $COMMON -b1_eps 0.04 -pc_sor_forward -ksp_monitor_true_residual &> $OUT
OUT=both1gs_backward.txt
echo "generating $OUT ..."
../both1 $COMMON -b1_eps 0.04 -pc_sor_backward -ksp_monitor_true_residual &> $OUT
OUT=both1gs_ssor.txt
echo "generating $OUT ..."
../both1 $COMMON -b1_eps 0.04 -ksp_monitor_true_residual &> $OUT

