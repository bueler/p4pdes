#!/bin/bash
set -e
set +x

# generate matrix .dat files for script p4pdes-book/figs/schureigs.py
#   which generates a figure in chapter 13 of book; the goal is to show
#   dependence of norms of S_h^{-1} on the mesh size h, for two different
#   mixed FE methods
# here we just write the whole system matrix K_h
# note the problem is -nobase so that K_h has no kernel

# set rtol to 1.0 so succeeds on first iter
SOLVE="-s_ksp_type minres -s_pc_type none -s_ksp_rtol 1.0"

#FIXME what elements do I want?

for LEV in 1 2 3 4; do  # LEV=4 is already generating N=O(10^4) matrices
    OUT=schur_P2P1_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 2 -pdegree 1 \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
    OUT=schur_CD_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 2 -pdegree 0 -dpressure \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
done

