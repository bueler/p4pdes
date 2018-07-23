#!/bin/bash
set -e
set +x

# FIXME make figure with h/p-versus-error
# FIXME make dual-axis figure with h/p-versus-matrix-dimension and h/p-versus-sparsity-ratio

# run as
#    ./hprefine.sh &> hprefine.txt

function runcase() {
    CMD="../firefish.py -mx $1 -my $1 -order $2 $3 -s_ksp_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "done on" tmp.txt
    grep "error " tmp.txt
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "rows"
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "total:"
}

# h refine
for M in 5 9 17 33 65 129 257 513 1025; do
     runcase $M 1 "-s_ksp_type cg -s_pc_type gamg -s_ksp_rtol 1.0e-14"
done

# p refine
for P in 1 2 3 4 5 6 7; do
     runcase 5 $P "-s_ksp_type preonly -s_pc_type lu"
done

rm -rf tmp.txt

