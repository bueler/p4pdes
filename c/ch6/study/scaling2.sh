#!/bin/bash
set -e

# run as:   ./scaling2.sh &> scaling2.txt

# use PETSC_ARCH with --with-debugging=0

# going to -da_refine 12 gives out of memory on WORKSTATION with 16Gb

# shows number of CG iterations constant and times which scale
# linearly with number of unknowns, and error which is O(h^2):
#   (level)    (mx)      (num of KSP iters)      (time)
#   6          129       5                       0.11
#   7          257       5                       0.25
#   8          513       5                       0.84
#   9          1025      5                       2.93
#   10         2049      5                       10.97
#   11         4097      4                       38.41

# >> mx = 2.^(7:12)+1
# >> time = [0.11 0.25 0.84 2.93 10.97 38.41]
# >> polyfit(log(mx(3:end).^2),log(time(3:end)),1)
# ans =
#      0.92322     -11.704
# ... slightly sublinear

for LEV in 6 7 8 9 10 11; do
    CMD="../fish2 -da_refine $LEV -ksp_type cg -pc_type mg -ksp_converged_reason -ksp_rtol 1.0e-10 -snes_max_it 1"
    echo "COMMAND:  $CMD"
    /usr/bin/time --portability -f "real %e" $CMD
done

