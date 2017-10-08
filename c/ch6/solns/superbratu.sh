#!/bin/bash
set -e

# summary:   Solve Bratu equation by FAS using NGS smoothing.  The coarse grid
# is 17x17 and solved by Newton-Krylov using CG+ICC.

# use --with-debugging=0 build for timing

# highest res 8193x8193 grid uses 13.5 GB on WORKSTATION

# add to see what is happening: 
#     -snes_monitor -snes_fas_monitor -fas_coarse_snes_monitor -fas_coarse_ksp_converged_reason

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 6 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 4

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 7 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 5

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 6

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 9 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 7

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 10 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 8

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 11 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 9

/usr/bin/time --portability -f "real %e"  ./bratu2D -snes_converged_reason -lb_showcounts -da_refine 12 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc  -snes_fas_levels 10


#$ ./superbratu.sh 
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 6.431e+07,  residual calls = 390,  NGS calls = 60
#done on 129 x 129 grid ...
#real 0.20
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 1.663e+08,  residual calls = 443,  NGS calls = 80
#done on 257 x 257 grid ...
#real 0.47
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 5.689e+08,  residual calls = 495,  NGS calls = 100
#done on 513 x 513 grid ...
#real 1.61
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 2.157e+09,  residual calls = 545,  NGS calls = 120
#done on 1025 x 1025 grid ...
#real 5.27
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 8.398e+09,  residual calls = 595,  NGS calls = 140
#done on 2049 x 2049 grid ...
#real 18.86
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 10
#flops = 3.285e+10,  residual calls = 645,  NGS calls = 160
#done on 4097 x 4097 grid ...
#real 70.75
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 11
#flops = 1.388e+11,  residual calls = 741,  NGS calls = 198
#done on 8193 x 8193 grid ...
#real 290.10

