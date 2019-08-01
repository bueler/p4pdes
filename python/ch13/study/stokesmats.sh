#!/bin/bash
set -e
set +x

# generate .dat files, i.e. matrices, for script p4pdes-book/figs/stokeseigs.py
# (it generates a figure in chapter 13 of book)

SOLVE="-s_ksp_type gmres  -s_ksp_converged_reason -s_ksp_rtol 1.0e-8 -pcpackage schur_lower_gmg"

for MU in 0.01 0.1 1.0 10.0 100.0; do
    ../stokes.py -mu $MU -refine 1 $SOLVE \
        -s_mat_type aij -s_ksp_view_mat binary:stokesmat${MU}.dat
done

