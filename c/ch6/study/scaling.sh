#!/bin/bash
set -e

# shows number of CG iterations constant and times (on laptop) which scale
# linearly with number of unknowns, and error which is O(h^2):
#   (level)    (num of unknowns)      (num of KSP iters)      (time)
#   6          16641                  5                       0.11
#   7          66049                  5                       0.32
#   8          263169                 5                       1.07
#   9          1050625                5                       3.74
#   10         4198401                5                       14.24
#   11         16785409               4                       51.01

# fit:  time = O(unknowns^0.92)   so slightly sublinear

#FIXME: redo on workstation

# use PETSC_ARCH with --with-debugging=0

for LEV in 6 7 8 9 10 11; do
    CMD="../fish2 -da_refine $LEV -ksp_type cg -pc_type mg -ksp_converged_reason -ksp_rtol 1.0e-10 -snes_max_it 1"
    echo "COMMAND:  $CMD"
    /usr/bin/time --portability -f "real %e" $CMD
done

