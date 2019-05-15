#!/bin/bash

# Generate data to measure flop/N for ad3 using GMG with ILU smoothing on
# advection-diffusion problem.  Uses no limiter (= first-order upwinding) and
# small coarse grid for efficiency.

# Results (mx=mz so N=mx^2*my):
# mx  my  flop       time (s)   flop/N
# 13  12  1.826e+06  2.211e-02   900.4
# 25  24  1.893e+07  1.374e-01  1262.0
# 49  48  1.861e+08  1.038e+00  1614.8
# 97  96  1.464e+09  7.624e+00  1620.8
#193 192  1.161e+10  4.674e+01  1623.4

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


