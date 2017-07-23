#!/bin/bash
set -e

EXEC=$1
REFINE=$2

# use PETSC_ARCH with --with-debugging=0
# run as in these examples:
#    $ ./mgstudy.sh "../fish -fsh_dim 2" 9 &> fish2refine9.txt
#    $ ./mgstudy.sh "../fish -fsh_dim 3" 5 &> fish3refine5.txt
#    $ ./mgstudy.sh "mpiexec -n 2 ../fish -fsh_dim 2" 5 &> fish2refine5mpi2.txt

# currently does (REFINE/2) * 2 * 2 * 3 = 6*REFINE runs

# multigrid parameter study on fixed-size problem using executable EXEC:
#     -pc_mg_levels             <REFINE+1|REFINE-1|...|3>
#     -pc_mg_cycle_type         <v|w>
#     -mg_levels_ksp_max_it     <1|2|3>                  HOW MANY SMOOTHINGS (UP AND DOWN)

# WHICH SMOOTHER:
#     -mg_levels_ksp_type       <chebyshev|richardson>

# for Richardson smoother, SSOR is default, but we COULD compare SSOR, GS, and alpha=0.8 Jacobi:
#     -mg_levels_pc_type sor                                  # SSOR: fewest iters, fastest
#     -mg_levels_pc_type sor -mg_levels_pc_sor_forward        # GS: more iters, almost as fast
#     -mg_levels_pc_type jacobi -mg_levels_ksp_richardson_scale 0.8  # weighted Jacobi: more iters, slower (but still much better than unweighted)

for (( MGLEV=$(( REFINE + 1 )); MGLEV>2; MGLEV-=2 )); do
    for MGVW in v w; do
        for MGKSPTYPE in chebyshev richardson; do
            for MGSMOOTHIT in 1 2 3; do
                CMD="$EXEC -snes_type ksponly -ksp_rtol 1.0e-12 -ksp_converged_reason -pc_type mg -da_refine $REFINE -pc_mg_levels $MGLEV -pc_mg_cycle_type $MGVW -mg_levels_ksp_type $MGKSPTYPE -mg_levels_ksp_max_it $MGSMOOTHIT"
                echo "COMMAND:  $CMD"
                /usr/bin/time --portability -f "real %e" $CMD
            done
        done
    done
done

