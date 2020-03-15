#!/bin/bash
set -e
set +x

# generate matrix .dat files to show dependence of norms of S_h^{-1} on the
# mesh size h, for different mixed FE methods; here we just write the whole
# system matrix K_h; the problem is -nobase so K_h is invertible IF the
# elements are stable

# see p4pdes-book/figs/schurinverse.py to process these

# set rtol to 1.0 so succeeds on first iter
SOLVE="-s_ksp_type minres -s_pc_type none -s_ksp_rtol 1.0"

for LEV in 1 2 3 4; do  # LEV=4 is generates N=O(10^4) matrices
    OUT=schur_P2P1_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 2 -pdegree 1 \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
    OUT=schur_Q2Q1_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -quad -udegree 2 -pdegree 1 \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
    OUT=schur_CD_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 2 -pdegree 0 -dp \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
    OUT=schur_P1P1_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 1 -pdegree 1 \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
    OUT=schur_P1P0_lev${LEV}.dat
    echo "generating ${OUT} ..."
    ../stokes.py -nobase $SOLVE -refine $LEV \
        -udegree 1 -pdegree 0 -dp \
        -s_mat_type aij -s_ksp_view_mat binary:${OUT}
done

