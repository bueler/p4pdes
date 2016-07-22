#!/bin/bash
set -e

# serial multigrid parameters study on fixed-size problem (4x10^6 or 10^6
# degrees freedom):
#     -pc_mg_levels             <REFINE|REFINE-2|...|3>
#     -pc_mg_cycle_type         <v|w>
#     -mg_levels_ksp_type       <chebyshev|cg|richardson>   WHICH SMOOTHER?
#     -mg_levels_ksp_max_it     <1|2|3>                     HOW MANY SMOOTHINGS (UP AND DOWN)

# use PETSC_ARCH with --with-debugging=0
# run as:
#    $ ./mgserialstudy.sh &> study.txt

REFINE=9 # or =10

for (( MGLEV=$REFINE; MGLEV>2; MGLEV-=2 )); do
    for MGVW in v w; do
        for MGSMOOTHTYPE in chebyshev cg richardson; do
            for MGSMOOTHIT in 1 2 3; do
                echo "case:  -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGSMOOTHTYPE -mg_levels_ksp_max_it $MGSMOOTHIT"
                /usr/bin/time --portability -f "real %e" ./fish2 -pc_type mg -da_refine $REFINE -ksp_rtol 1.0e-12 -ksp_converged_reason -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGSMOOTHTYPE -mg_levels_ksp_max_it $MGSMOOTHIT
            done
        done
    done
done

