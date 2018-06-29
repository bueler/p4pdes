#!/bin/bash

# Generate data to measure flop/N for ad3 using GMG with ILU smoothing on
# advection-diffusion problem.  Uses no limiter (= first-order upwinding) and
# small coarse grid for efficiency.

MAXLEV=6
EPS=0.1

for (( Z=2; Z<=$MAXLEV; Z++ )); do
    echo "running level ${Z}"
    cmd="../ad3 -ad3_eps ${EPS} -ad3_limiter none -da_grid_x 4 -da_grid_y 3 -da_grid_z 4 -da_refine ${Z} -ksp_type bcgs -ksp_rtol 1.0e-9 -ksp_monitor -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu"
    rm -f foo.txt
    echo $cmd
    $cmd -log_view &> foo.txt
    'grep' "numerical error:" foo.txt
    'grep' "grid:  " foo.txt
    'grep' "Flop:  " foo.txt
    'grep' "Time (sec):" foo.txt
done


