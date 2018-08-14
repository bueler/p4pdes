#!/bin/bash
set -e
set +x

# run as
#    ./stokesconv.sh &> stokesconv.txt

# FE methods: P2xP1, Q2xQ1, P3xP2, CD(P2xP0)

for FE in "" "-quad" "-udegree 3 -pdegree 2" "-pdegree 0 -dpressure"; do
    for LEV in 2 3 4 5 6 7 8; do  # WARNING: level 8 (=513x513) is expensive
        ../stokes.py -analytical -package schur2 -s_ksp_converged_reason \
            -s_ksp_rtol 1.0e-11 $FE -refine $LEV
    done
done

