#!/bin/bash

#set -e   # don't halt if one run fails
set +x

# run with --with-debugging=0 build
# run as
#    ./limjac.sh
# and hand-generate table in book

EXEC=../advect
LEV=5    # results (on number of Newton iterations) seem independent of this(?)

# note initial condition (stump|smooth|cone|box) irrelevant because none|centered
# Jacobians are not affected 

for LIM in none centered vanleer koren; do
    for JAC in none centered; do
        echo "limiter=" $LIM ", jacobian=" $JAC ":"
        $EXEC -da_refine $LEV -ts_dt 0.01 -ts_final_time 0.01 \
           -ts_type cn -ts_monitor -snes_converged_reason -ksp_rtol 1.0e-12 \
            -adv_limiter $LIM -adv_jacobian $JAC -snes_max_it 200
        echo
    done
done

