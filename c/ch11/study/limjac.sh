#!/bin/bash

#set -e   # don't halt if one run fails
set +x

# run with --with-debugging=0 build
# run as
#    ./limjac.sh &> limjac.txt
# and hand-generate table in book

LEV=4
DT=0.01  # for LEV=4, CFL of 0.5 gives dt = 0.00625

# using stump initial
for LIM in none centered vanleer koren; do
    for JAC in none centered; do
        echo "limiter=" $LIM ", jacobian=" $JAC ":"
        ../advect -da_refine $LEV -ts_dt $DT -ts_final_time $DT -ts_type cn \
             -ksp_rtol 1.0e-12 -snes_converged_reason -snes_max_it 200 \
             -adv_limiter $LIM -adv_jacobian $JAC
        echo
    done
done

