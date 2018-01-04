#!/bin/bash
set -e

# study condition numbers of preconditioned Poisson operators

# run as:
#   ./pccondition.sh &> pccondition.txt
# use PETSC_ARCH with --with-debugging=0

# results & figure-generation:  see p4pdes-book/figs/pccondition.txt|py

for LEV in 3 4 5 6 7 8 9; do
    for PC in none icc mg; do
        CMD="../fish -ksp_monitor_singular_value -da_refine $LEV -pc_type $PC"
        #echo "COMMAND:  $CMD"
        rm -rf tmp.txt
        $CMD &> tmp.txt
        COND=`tail -n 3 tmp.txt | head -n 1 | awk '{print $12}'`
        echo "$LEV $COND # $PC"
    done
done

