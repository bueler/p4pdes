#!/bin/bash
set -e
set +x

# run as
#    ./stokesconv.sh &> stokesconv.txt

COMMON="-analytical -s_ksp_rtol 1.0e-8 -s_ksp_converged_reason"

# some GMG+Schur solver alternatives use one each of the following:
#    -s_ksp_type gmres|fgmres|minres
#    -schurgmg diag|lower|full
#    -schurpre selfp|mass
SOLVE="-s_ksp_type gmres -schurgmg lower -schurpre selfp"

# FE methods are: P2xP1, Q2xQ1, P2xP0 (CD), P3xP2
# level 6 is 129x129, 7 is 257x257, 8 is 513x513, 9 is 1025x1025
for FE in "" "-quad" "-pdegree 0 -dp" "-udegree 3 -pdegree 2"; do
    for LEV in 2 3 4 5 6 7 8; do   # add levels as desired
        cmd="../stokes.py ${COMMON} ${SOLVE} ${FE} -refine ${LEV}"
        echo $cmd
        /usr/bin/time -f "real %e" $cmd
    done
done

