#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build
# run as
#    ./verif.sh &> advect-workstation.txt
# and copy the .txt file to figs/ in book repo for figure generation
# see p4pdes-book/figs/advect-conv.py

NPROCS=8
EXEC=../advect
LAPS=1            # could run additional experiment with e.g. 5 laps

for LEV in 3 4 5 6 7 8; do
    for SHAPE in stump smooth cone; do # "box" is other possibility, but doesnt add much over stump
        for LIMITER in centered none vanleer koren; do
            mpiexec -n $NPROCS $EXEC -adv_oneline \
                -ts_final_time $LAPS -da_refine $LEV \
                -adv_limiter $LIMITER -adv_initial $SHAPE
        done
    done
done
