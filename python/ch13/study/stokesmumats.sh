#!/bin/bash
set -e
set +x

# generate matrix .dat files for script p4pdes-book/figs/stokesmueigs.py
#   which generates a figure in chapter 13 of book
# shows dependence of eigenvalues on the viscosity mu
# note the problem is -nobase so that there is no kernel

SOLVE="-s_ksp_type gmres -s_ksp_converged_reason -s_ksp_rtol 1.0e-8 -schurgmg lower_mass"

# -refine 2 is 9x9 grid
for MU in 0.00001 0.0001 0.001 0.01 0.1 1.0 10.0 100.0; do
    echo "mu = ${MU}:"
    ../stokes.py -nobase -mu $MU -refine 2 $SOLVE \
        -s_mat_type aij -s_ksp_view_mat binary:stokes_mu${MU}.dat
done
