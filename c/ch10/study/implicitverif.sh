#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build
# run as
#    ./implicitverif.sh &> implicit-advect-workstation.txt
# and copy the .txt file to figs/ in book repo for figure generation  FIXME
# see p4pdes-book/figs/advect-conv.py

NPROCS=4
EXEC=../advect
LAPS=1

#FIXME once -snes_mf_operator is in master, add timing

#for LEV in 3 4 5 6 7 8; do
for LEV in 3 4 5 6; do
    for LIMITER in none vanleer; do
        mpiexec -n $NPROCS $EXEC -adv_oneline -ts_type cn \
            -ts_final_time $LAPS -da_refine $LEV \
            -adv_limiter $LIMITER -adv_jacobian none -adv_initial smooth #-snes_converged_reason
    done
    # FIXME: add vanleer + -snes_mf_operator
    mpiexec -n $NPROCS $EXEC -adv_oneline -ts_type cn \
        -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter centered -adv_jacobian centered -adv_initial smooth #-snes_converged_reason
done
