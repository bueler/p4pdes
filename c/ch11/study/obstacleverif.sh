#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build, as
#   ./obstacleverif.sh &> obstacleverif.txt
# generates table in chapter 11

MAXLEV=8

for GRD in -da_refine -snes_grid_sequence; do
    for TYPE in vinewtonrsls vinewtonssls; do
        echo
        echo "grid by ${GRD}, type ${TYPE}:"
        for (( LEV=3; LEV<=$MAXLEV; LEV++ )); do
            ../obstacle -snes_type $TYPE $GRD $LEV -ksp_type cg -pc_type mg -snes_max_it 200;
        done;
    done
done

