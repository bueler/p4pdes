#!/bin/bash
set -e

# run as:
#   ./scaling3.sh &> scaling3.txt
# use PETSC_ARCH with --with-debugging=0
# limitations: going to -da_refine 8 gives out of memory on WORKSTATION with 16Gb
# compare scaling2.sh

# re measuring timing:  the time in SNESSolve includes assembly of system on
# each coarser grid, through calls to FormJacobianLocal(); note
# FormFunctionLocal() only called once, on finest grid

# shows number of CG iterations constant (7 at each level) and error is O(h^2):
#   (level)    (mx)      (time)      (flops)
#   3          17        2.8710e-02  1.31e+07
#   4          33        1.5960e-01  9.65e+07
#   5          65        9.2471e-01  7.41e+08
#   6          129       6.1326e+00  5.81e+09
#   7          257       4.6442e+01  4.60e+10

# copy block of data above into data3.txt; then do in Matlab (FIXME: python):
# >> load('data3.txt')
# >> mx = data3(:,2);  time = data3(:,3);  flops = data3(:,4);
# >> pt = polyfit(log(mx.^3),log(time),1)
# pt =
#       0.9213     -11.558                     % slightly sublinear
# >> loglog(mx,time,'k-o',mx,exp(pt(1)*log(mx.^3)+pt(2)),'k--')
# >> pf = polyfit(log(mx.^3),log(flops),1)
# pf =
#       1.0015      7.8809                     % perfectly linear

for LEV in 3 4 5 6; do
#for LEV in 3 4 5 6 7; do
    CMD="../fish -fsh_dim 3 -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

