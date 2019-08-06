#!/bin/bash
set -e
set +x

# run as
#    ./stokesweak.sh &> stokesweak.txt

# problem is default lid-driven cavity with Dirichlet on whole boundary
# FE method is Q^2 x Q^1 Taylor-Hood

SOLVE="-s_ksp_type gmres -schurgmg lower_nomass"  # see stokesopt.sh for reason for this solver

# COARSE and LEV imply 129x129 grid on each process
COARSE="-mx 9 -my 9"
LEV0=4

LEV=$LEV0
P=1
for X in 1 2 3 4; do
    cmd="mpiexec -n ${P} ../stokes.py -quad -showinfo -s_ksp_converged_reason ${SOLVE} ${COARSE} -refine ${LEV} -log_view"
    echo $cmd
    rm -f foo.txt
    $cmd &> foo.txt
    'grep' "sizes:" foo.txt
    'grep' "solve converged due to" foo.txt
    'grep' "Flop:  " foo.txt
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
    P=$(( $P * 4 ))
    LEV=$(( $LEV + 1 ))
done

