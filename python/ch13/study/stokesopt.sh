#!/bin/bash
set -e
set +x

# run as
#    ./stokesopt.sh &> stokesopt.txt

# problem is default lid-driven cavity with Dirichlet on whole boundary
# FE method is Q^2 x Q^1 Taylor-Hood

MAXLEV=9   # LEV=9 is 1025x1025 uniform grid with N about 10^7
for SOLVE in "-s_ksp_type minres -schurgmg diag" \
             "-s_ksp_type minres -schurgmg diag_nomass" \
             "-s_ksp_type gmres -schurgmg lower" \
             "-s_ksp_type gmres -schurgmg lower_nomass"; do      
    for (( LEV=2; LEV<=$MAXLEV; LEV++ )); do
        cmd="../stokes.py -quad -showinfo -s_ksp_converged_reason ${SOLVE} -refine ${LEV} -log_view"
        echo $cmd
        rm -f foo.txt
        $cmd &> foo.txt
        'grep' "sizes:" foo.txt
        'grep' "solve converged due to" foo.txt
        'grep' "Flop:  " foo.txt | awk '{print $2}'
        'grep' "Time (sec):" foo.txt | awk '{print $3}'
    done
done

