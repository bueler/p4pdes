#!/bin/bash
set -e
set +x

# run as
#    ./stokesconv.sh &> stokesconv.txt

# FE methods are: P2xP1, Q2xQ1, P3xP2, CD(P2xP0)

for FE in "" "-quad" "-udegree 3 -pdegree 2" "-pdegree 0 -dpressure"; do
    for SOLVE in "-s_ksp_type gmres -pcpackage schur_lower_gmg" \
                 "-s_ksp_type fgmres -pcpackage schur_lower_gmg_nomass" \
                 "-s_ksp_type gmres -pcpackage schur_diag_gmg"; do
        #for LEV in 2 3 4 5 6 7 8; do  # WARNING: level 8 (=513x513) is expensive
        for LEV in 3 4 5 6; do  # quicker
            /usr/bin/time -f "real %e" ../stokes.py -analytical $FE -refine $LEV $SOLVE \
                -s_ksp_converged_reason -s_ksp_rtol 1.0e-8
        done
    done
done

