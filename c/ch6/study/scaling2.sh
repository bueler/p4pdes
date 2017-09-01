#!/bin/bash
set -e

# run as:
#   ./scaling2.sh &> scaling2.txt
# use PETSC_ARCH with --with-debugging=0
# limitation: going to -da_refine 12 gives out of memory on WORKSTATION with 16Gb
# compare scaling3.sh

# re measuring timing:  the time in SNESSolve includes assembly of system on
# each coarser grid, through calls to FormJacobianLocal(); note
# FormFunctionLocal() only called once, on finest grid

# shows number of CG iterations constant (7 at each level) and error is O(h^2):
#   (level)    (mx)      (time)      (flops)
#   6          129       9.2926e-02  4.04e+07
#   7          257       2.4581e-01  1.60e+08
#   8          513       9.0780e-01  6.39e+08
#   9          1025      3.1085e+00  2.55e+09
#   10         2049      1.1834e+01  1.02e+10
#   11         4097      4.5728e+01  4.08e+10

# copy block of data above into data2.txt; then:
# >> load('data2.txt')
# >> mx = data2(:,2);  time = data2(:,3);  flops = data2(:,4);
# >> pt = polyfit(log(mx.^2),log(time),1)
# pt =
#      0.90554     -11.338                     % slightly sublinear
# >> loglog(mx,time,'k-o',mx,exp(pt(1)*log(mx.^2)+pt(2)),'k--')
# >> pf = polyfit(log(mx.^2),log(flops),1)
# pf =
#       1.0003      7.7907                     % perfectly linear

for LEV in 6 7 8 9 10 11; do
    CMD="../fish -fsh_dim 2 -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

