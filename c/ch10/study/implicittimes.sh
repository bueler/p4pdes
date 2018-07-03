#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build
# run as
#    ./implicittimes.sh &> implicit-advect.txt

EXEC=../advect
LAPS=1.0
LEV=4

# CN + (correct jacobian for none|centered limiter)
for LIMITER in none centered; do
    time $EXEC -adv_oneline -ts_type cn \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter $LIMITER -adv_jacobian $LIMITER
done
# CN + (vanleer limiter) + (use none Jacobian two ways)
for JFNK in "" "-snes_mf_operator"; do
    time $EXEC -adv_oneline -ts_type cn \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter vanleer -adv_jacobian none $JFNK
done
# RK
for LIMITER in none centered vanleer; do
    time $EXEC -adv_oneline \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter $LIMITER
done


