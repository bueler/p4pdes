#!/bin/bash
set -e

NPROCS=4
EXEC=../advect
LAPS=1

# run with --with-debugging=0 build

for LEV in 2 3 4 5; do
    for SHAPE in box stump cone; do
        for LIMITER in centered none vanleer koren; do
            echo
            mpiexec -n $NPROCS $EXEC -ts_final_time $LAPS -da_refine $LEV \
                -adv_limiter $LIMITER -adv_initial $SHAPE
        done
    done
done
