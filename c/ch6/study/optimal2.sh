#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 2D

# run as:
#   ./optimal2.sh &> optimal2.txt

# use PETSC_ARCH with --with-debugging=0

# limitation: going to -da_refine 12 gives out of memory on WORKSTATION with 16Gb

# results:  see p4pdes-book/figs/optimal2.txt

# copy block of data above into data2.txt; then:   # FIXME: replace w python
# >> load('data2.txt')
# >> mx = data2(:,2);  time = data2(:,3);  flops = data2(:,4);
# >> pt = polyfit(log(mx.^2),log(time),1)
# pt =
#      0.90554     -11.338                     % slightly sublinear
# >> loglog(mx,time,'k-o',mx,exp(pt(1)*log(mx.^2)+pt(2)),'k--')
# >> pf = polyfit(log(mx.^2),log(flops),1)
# pf =
#       1.0003      7.7907                     % perfectly linear

for LEV in 5 6 7 8 9 10 11; do
    CMD="../fish -fsh_dim 2 -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

