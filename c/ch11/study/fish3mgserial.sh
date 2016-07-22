#!/bin/bash
set -e

# serial multigrid parameter study on fixed-size problem:
#     -pc_mg_levels             <REFINE|REFINE-2|...|3>
#     -pc_mg_cycle_type         <v|w>
#     -mg_levels_ksp_type       <chebyshev|richardson>      WHICH SMOOTHER?  [cg fails or slow for fish3]
#     -mg_levels_ksp_max_it     <1|2|3>                     HOW MANY SMOOTHINGS (UP AND DOWN)

# use PETSC_ARCH with --with-debugging=0
# run as:
#    $ ./fish3mgserial.sh &> fish3.txt

REFINE=6 # or =5

for (( MGLEV=$REFINE; MGLEV>2; MGLEV-=1 )); do
    for MGVW in v w; do
        for MGSMOOTHTYPE in chebyshev richardson; do
            for MGSMOOTHIT in 1 2 3; do
                echo "case:  -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGSMOOTHTYPE -mg_levels_ksp_max_it $MGSMOOTHIT"
                /usr/bin/time --portability -f "real %e" ../fish3 -pc_type mg -da_refine $REFINE -ksp_rtol 1.0e-12 -ksp_converged_reason -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGSMOOTHTYPE -mg_levels_ksp_max_it $MGSMOOTHIT -ksp_max_it 200
            done
        done
    done
done

