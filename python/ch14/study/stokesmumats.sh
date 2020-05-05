#!/bin/bash
set -e
set +x

# generate matrix .dat files for script p4pdes-book/figs/stokesmueigs.py
# which generates a figure in chapter 14 of book showing dependence of
# eigenvalues on the viscosity mu

# note the problem is -nobase so there is no kernel

SOLVE="-s_ksp_type minres -s_pc_type none" # no preconditioning; want eigs of K not M^-1 K
CONVERGE="-s_ksp_converged_reason -s_ksp_rtol 1.0e0"  # converges on first iteration
LEV=2   # 9x9 grid

for MU in 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 100.0; do
    echo "mu = ${MU}:"
    ../stokes.py -nobase -mu $MU -refine $LEV $SOLVE $CONVERGE \
        -s_mat_type aij -s_ksp_view_mat binary:stokes_mu${MU}.dat
done
