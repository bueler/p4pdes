#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build, as
#   ./obstaclesolvers.sh &> obstaclesolvers.txt
# generates tables 12.1 and 12.2

MAXLEV=7

for GRD in -da_refine -snes_grid_sequence; do
    for TYPE in vinewtonrsls vinewtonssls; do
        echo
        echo "grid by ${GRD}, type ${TYPE}:"
        for (( LEV=3; LEV<=$MAXLEV; LEV++ )); do
            ../obstacle -snes_type $TYPE $GRD $LEV -ksp_type cg -pc_type mg -snes_max_it 200 -mg_levels_ksp_max_it 3;
        done;
    done
done

