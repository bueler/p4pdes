#!/bin/bash
set -e
set +x

# run as
#    ./stokesconv.sh &> stokesconv.txt

# methods: P2xP1, Q2xQ1, P3xP2, CD(P2xP0)

for METH in "" "-quad" "-uorder 3 -porder 2" "-porder 0 -dpressure"; do
    for L in 2 3 4 5 6 7 8; do  # WARNING: level 8 expensive
        ../stokes.py -analytical $METH -refine $L -s_ksp_rtol 1.0e-11
    done
done

