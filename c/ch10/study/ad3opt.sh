#!/bin/bash

# generate data to measure flop/N for ad3 using GMG with ILU smoothing on
# advection-diffusion problem

MAXLEV=5   # can go to level 6 on ed-galago ... uses ~ 36 Gb memory

for (( Z=2; Z<=$MAXLEV; Z++ )); do
    echo "running level ${Z}"
    cmd="../ad3 -ad3_limiter none -ksp_converged_reason -da_refine ${Z} -ksp_rtol 1.0e-9 -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -ksp_monitor -ad3_eps 0.1 -ksp_type bcgs"
    rm -f foo.txt
    echo $cmd
    $cmd -log_view &> foo.txt
    'grep' "numerical error:" foo.txt
    'grep' "grid:  " foo.txt
    'grep' "Flop:  " foo.txt
    'grep' "Time (sec):" foo.txt
done


