#!/bin/bash
set -e
set +x

# run as
#    ./hprefine.sh &> hprefine.txt

function runcase() {
    CMD="../fish.py -refine $1 -order $2 $3 -s_ksp_view -log_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "done on" tmp.txt
    grep "error " tmp.txt
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "rows"
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "total:"
    grep "Flop:    " tmp.txt
}

# h refine
for L in 1 2 3 4 5 6 7 8 9; do
     runcase $L 1 "-s_ksp_type cg -s_pc_type mg -s_ksp_rtol 1.0e-14"
done

# p refine
for P in 1 2 3 4 5 6 7 8; do
     runcase 1 $P "-s_ksp_type preonly -s_pc_type cholesky"
done

rm -rf tmp.txt

