#!/bin/bash
set -e

# considers performance, namely KSP iterations, of a CG+MG V cycles solver for
# the 3D Poisson equation using coarse grids of various resolutions, i.e.
# using V cycles of various depth, for the same coarse grid

# use PETSC_ARCH with --with-debugging=0

# see corresponding exercise in Chapter 6

MAXLEV=6     # =6 corresponds to 129x129x129 grid

# the range of depths (multigrid levels) of the V cycles
STARTDEPTH=7   # = $(( $MAXLEV + 1 ))
ENDDEPTH=3     # going to 2 essentially stalls

# default CSOLVE: use LU with n.d. variable ordering on the coarse grid
#   ---> conclude flops jump up when depth is small (e.g. 4 or 3 if MAXLEV=6)
# other CSOLVE: use 2 iterations of preconditioned CG on the coarse grid
#   ---> conclude number of V cycles (KSP iterations) increase when depth is small (again: 4 or 3)
for CSOLVE in "" "-mg_coarse_ksp_type cg -mg_coarse_pc_type icc -mg_coarse_ksp_max_it 2 -mg_coarse_ksp_converged_reason -mg_coarse_ksp_convergence_test skip"; do
    for (( DEPTH=$STARTDEPTH; DEPTH>=$ENDDEPTH; DEPTH-- )); do
        CMD="../fish -fsh_dim 3 -ksp_rtol 1.0e-10 -ksp_converged_reason -pc_type mg -da_refine $MAXLEV -pc_mg_levels $DEPTH $CSOLVE -log_view"
        echo "COMMAND:  $CMD"
        rm -rf tmp.txt
        $CMD &> tmp.txt
        #grep "solve converged" tmp.txt  see mg_coarse iterations to confirm
        grep -A 1 "Linear solve" tmp.txt
        grep "Flop:  " tmp.txt
    done
done

