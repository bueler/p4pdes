#!/bin/bash
set -e

# demonstrates optimality of a CG+MG solver for Poisson equation in 3D

# run as:
#   ./optimal3.sh &> optimal3.txt
# use PETSC_ARCH with --with-debugging=0

# limitations: going to -da_refine 8 gives out of memory on WORKSTATION with 16Gb

# re measured timing:  the time in SNESSolve includes assembly of system on
# each coarser grid (= calls to FormJacobianLocal()), but note
# FormFunctionLocal() is only called once (on finest grid)

# results:  see p4pdes-book/figs/optimal3.txt

# copy block of data above into data3.txt; then do in Matlab (FIXME: replace with python):
# >> load('data3.txt')
# >> mx = data3(:,2);  time = data3(:,3);  flops = data3(:,4);
# >> pt = polyfit(log(mx.^3),log(time),1)
# pt =
#       0.9213     -11.558                     % slightly sublinear
# >> loglog(mx,time,'k-o',mx,exp(pt(1)*log(mx.^3)+pt(2)),'k--')
# >> pf = polyfit(log(mx.^3),log(flops),1)
# pf =
#       1.0015      7.8809                     % perfectly linear

for LEV in 3 4 5 6 7; do
    CMD="../fish -fsh_dim 3 -snes_type ksponly -ksp_type cg -pc_type mg -ksp_rtol 1.0e-10 -ksp_converged_reason -da_refine $LEV -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep -C 1 "problem manuexp" tmp.txt
    grep SNESSolve tmp.txt
done

