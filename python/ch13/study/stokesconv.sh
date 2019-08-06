#!/bin/bash
set -e
set +x

# run as
#    ./stokesconv.sh &> stokesconv.txt

# minimal testing suggests this is faster than:
#    GMRES+lower_mass, MINRES+diag_mass, MINRES+diag
SOLVE="-s_ksp_type gmres -schurgmg lower"
COMMON="-analytical -s_ksp_rtol 1.0e-8 -s_ksp_converged_reason $SOLVE"

# FE methods are: P2xP1, Q2xQ1, P3xP2, CD(P2xP0)
# level 8 is 513x513
# level 9 is 1025x1025
for FE in "" "-quad" "-pdegree 0 -dpressure"; do
    for LEV in 2 3 4 5 6 7 8 9; do
        /usr/bin/time -f "real %e" ../stokes.py $COMMON $FE -refine $LEV
    done
done
# higher-degree is more memory per element
for FE in "-udegree 3 -pdegree 2"; do
    for LEV in 2 3 4 5 6 7 8; do
        /usr/bin/time -f "real %e" ../stokes.py $COMMON $FE -refine $LEV
    done
done

