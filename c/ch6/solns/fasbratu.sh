#!/bin/bash
set -e

# summary:   Solve Bratu equation by FAS using NGS smoothing.  The coarse grid
# is 17x17 and solved by Newton-Krylov using CG+ICC.

# use --with-debugging=0 build for timing

# highest res 8193x8193 grid uses 13.5 GB on WORKSTATION

# add to see what is happening: 
#     -snes_monitor -snes_fas_monitor -fas_coarse_snes_monitor -fas_coarse_ksp_converged_reason
# add to use exact solution:
#     -lb_exact -snes_rtol 1.0e-10
# also try:
#     -snes_fas_type multiplicative

OPTIONS="-snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc"

/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 6 -snes_fas_levels 2
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 7 -snes_fas_levels 3
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 8 -snes_fas_levels 4
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 9 -snes_fas_levels 5
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 10 -snes_fas_levels 6
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 11 -snes_fas_levels 7
/usr/bin/time --portability -f "real %e"  ./bratu2D $OPTIONS -da_refine 12 -snes_fas_levels 8

#$ ./superbratu.sh 
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 8
#flops = 1.317e+09,  residual calls = 304,  NGS calls = 24
#done on 129 x 129 grid ...
#real 0.97
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 6
#flops = 1.548e+09,  residual calls = 375,  NGS calls = 42
#done on 257 x 257 grid ...
#real 1.67
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 6
#flops = 1.811e+09,  residual calls = 458,  NGS calls = 78
#done on 513 x 513 grid ...
#real 2.59
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
#flops = 3.085e+09,  residual calls = 551,  NGS calls = 105
#done on 1025 x 1025 grid ...
#real 5.47
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
#flops = 7.979e+09,  residual calls = 703,  NGS calls = 155
#done on 2049 x 2049 grid ...
#real 16.22
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
#flops = 2.706e+10,  residual calls = 888,  NGS calls = 215
#done on 4097 x 4097 grid ...
#real 55.56
#Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
#flops = 1.015e+11,  residual calls = 1093,  NGS calls = 285
#done on 8193 x 8193 grid ...
#real 204.52


