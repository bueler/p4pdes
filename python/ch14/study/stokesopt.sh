#!/bin/bash
set -e
set +x

# run as
#    ./stokesopt.sh &> stokesopt.txt

# problem is default lid-driven cavity with Dirichlet on whole boundary
# FE method is Q^2 x Q^1 Taylor-Hood

MAXLEV=10   # LEV=9 is 1025x1025 uniform grid with N~~10^7, LEV=10 is 2049^2

# compare 6 Schur+GMG solvers
for SGMG in "-s_ksp_type minres -schurgmg diag" \
            "-s_ksp_type gmres -schurgmg lower" \
            "-s_ksp_type gmres -schurgmg full"; do
    for SPRE in "-schurpre selfp" \
                "-schurpre mass"; do
        for (( LEV=2; LEV<=$MAXLEV; LEV++ )); do
            cmd="../stokes.py -quad -showinfo -s_ksp_converged_reason ${SGMG} ${SPRE} -refine ${LEV} -log_view"
            echo $cmd
            rm -f foo.txt
            $cmd &> foo.txt
            'grep' "sizes:" foo.txt
            'grep' "solution norms:" foo.txt
            'grep' "solve converged due to" foo.txt
            'grep' "Flop:  " foo.txt | awk '{print $2}'
            'grep' "Time (sec):" foo.txt | awk '{print $3}'
        done
        echo
    done
done

