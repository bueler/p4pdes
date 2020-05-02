#!/bin/bash
set -e
set +x

# run as
#    ./stokesweak.sh &> stokesweak.txt

# problem is default lid-driven cavity with Dirichlet on whole boundary
# FE method is Q^2 x Q^1 Taylor-Hood

MPI="mpiexec --map-by core --bind-to hwthread"  # one possible setting

# see text of chapter for evidence this is a reasonable choice
SOLVE="-s_ksp_type gmres -schurgmg lower -schurpre selfp"

COARSE="-mx 9 -my 9" # need at least one point per process on coarse grid
LEV0=5  # 5 is 257x257 grid on each process

LEV=$LEV0
P=1
for X in 1 2 3 4; do   # 1->1, 2->4, 3->16, 4->64
    cmd="${MPI} -n ${P} ../stokes.py -quad -showinfo -s_ksp_converged_reason ${SOLVE} ${COARSE} -refine ${LEV} -log_view"
    echo $cmd
    rm -f foo.txt
    $cmd &> foo.txt
    'grep' "solving on" foo.txt
    'grep' "sizes:" foo.txt
    'grep' "solve converged due to" foo.txt
    'grep' "Flop:  " foo.txt
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
    P=$(( $P * 4 ))
    LEV=$(( $LEV + 1 ))
done

