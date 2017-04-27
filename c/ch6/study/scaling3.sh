#!/bin/bash
set -e

# run as:   ./scaling3.sh &> scaling3.txt

# use PETSC_ARCH with --with-debugging=0

# going to -da_refine 8 gives out of memory on WORKSTATION with 16Gb

# see scaling2.sh

# >> mx = 2.^(5:8)+1
# >> kspiters = [10 14 20 26]
# >> time = [0.23 1.70 13.65 128.89]
# >> polyfit(log(mx.^3),log(time),1)
# ans =
#      1.0264     -12.284
# ... pretty damn linear

# >> kspitersGAMG = [8 9 10 10]     # -pc_type gamg
# >> timeGAMG = [0.36 2.23 17.73 145.63]
# >> polyfit(log(mx.^3),log(timeGAMG),1)
# ans =
#      0.97829     -11.356

for LEV in 4 5 6 7; do
    CMD="../fish3 -da_refine $LEV -ksp_type cg -pc_type mg -ksp_converged_reason -ksp_rtol 1.0e-10 -snes_max_it 1"
    echo "COMMAND:  $CMD"
    /usr/bin/time --portability -f "real %e" $CMD
done

