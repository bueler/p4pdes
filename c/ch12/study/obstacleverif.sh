#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build, as
#   ./obstacleverif.sh &> obstacleverif.txt

# see p4pdes-book/figs/obstacle.py|txt for visualization

# defaults to RSLS solver

MAXLEV=10   # to 2049x2049 grid

for (( LEV=3; LEV<=$MAXLEV; LEV++ )); do
    rm -rf tmp.txt
    ../obstacle -snes_grid_sequence $LEV -ksp_type cg -pc_type mg -mg_levels_ksp_type richardson -log_view > tmp.txt
    grep "errors:" tmp.txt
    grep "last KSP iters" tmp.txt
    grep "Flop:  " tmp.txt
done;

