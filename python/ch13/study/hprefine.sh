#!/bin/bash
set -e
set +x

# FIXME this version only has p-refinement

# FIXME make figure with p-versus-error
# FIXME make dual-axis figure with p-versus-matrix-dimension and p-versus-sparsity-ratio

# FIXME figures as above include h-refine and hp-refine too?

# run as
#    ./hprefine.sh &> hprefine.txt

function runcase() {
    CMD="../firefish.py -mx $1 -my $1 -order $2 $3 -s_ksp_type preonly -s_pc_type lu -s_ksp_view"
    echo "COMMAND:  $CMD"
    rm -rf tmp.txt
    $CMD &> tmp.txt
    grep "done on" tmp.txt
    grep "error " tmp.txt
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "rows"
    grep -A 3 "Mat Object: (s_)" tmp.txt | grep "total:"
}

for P in 1 2 3 4 5 6 7; do
     runcase 5 $P ""
done
rm -rf tmp.txt

