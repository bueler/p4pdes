#!/bin/bash
set -e

EXEC=$1
REFINE=$2

# serial multigrid parameter study on fixed-size problem using executable EXEC:
#     -pc_mg_levels             <REFINE+1|REFINE-1|...|3>
#     -pc_mg_cycle_type         <v|w>
#     -mg_levels_ksp_type       <chebyshev|richardson>   WHICH SMOOTHER?
#     -mg_levels_ksp_max_it     <1|2|3>                  HOW MANY SMOOTHINGS (UP AND DOWN)
#     -mg_levels_pc_type        <jacobi|sor>

# use PETSC_ARCH with --with-debugging=0
# run as:
#    $ ./mgstudy.sh ../fish2 9 &> fish2refine9.txt
#    $ ./mgstudy.sh ../fish3 5 &> fish3refine5.txt

for (( MGLEV=$(( REFINE + 1 )); MGLEV>2; MGLEV-=2 )); do
    for MGVW in v w; do
        for MGKSPTYPE in chebyshev richardson; do
            for MGPCTYPE in jacobi sor; do
                for MGSMOOTHIT in 1 2 3; do
                    CMD="$EXEC -pc_type mg -ksp_rtol 1.0e-12 -ksp_converged_reason -da_refine $REFINE -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGKSPTYPE -mg_levels_pc_type $MGPCTYPE -mg_levels_ksp_max_it $MGSMOOTHIT"
                    echo "COMMAND:  $CMD"
                    /usr/bin/time --portability -f "real %e" $CMD
                done
            done
        done
    done
done

