#!/bin/bash
set -e
set +x

# demonstrate that for advect.c, implicit (CN) time stepping works but is much
# slower than explicit (RK)

# run with --with-debugging=0 build
# run as
#    ./implicittimes.sh &> implicittimes.txt

EXEC=../advect
LAPS=1.0
LEV=5

echo "CN + (correct jacobian)"
for LIMITER in none centered; do
    echo "limiter=$LIMITER"
    /usr/bin/time -f "real %e" $EXEC -adv_oneline -ts_type cn \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter $LIMITER -adv_jacobian $LIMITER
done
echo "CN + (vanleer limiter) + (none Jacobian)"
for JFNK in "" "-snes_mf_operator"; do
    echo "JFNK = $JFNK"
    /usr/bin/time -f "real %e" $EXEC -adv_oneline -ts_type cn \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter vanleer -adv_jacobian none $JFNK
done
echo "RK"
for LIMITER in none centered vanleer; do
    echo "limiter=$LIMITER"
    /usr/bin/time -f "real %e" $EXEC -adv_oneline \
        -adv_initial smooth -ts_final_time $LAPS -da_refine $LEV \
        -adv_limiter $LIMITER
done


