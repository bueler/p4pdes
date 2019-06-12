#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build, as
#   ./obstacleoptimal.sh &> obstacleoptimal.txt

# p4pdes-book/figs/obstacle.py uses result to generate N vs flops/N figure in Chapter 12

MAXLEV=10   # 10 is 2049^2 grid;  SS solver takes more than 20 min on this grid, so was stopped

for SOLVE in "-snes_type vinewtonrsls -pc_type mg -mg_levels_ksp_type richardson" \
             "-snes_type vinewtonrsls -pc_type gamg -pc_gamg_type classical" \
             "-snes_type vinewtonssls -pc_type gamg -pc_gamg_type classical"; do
    echo "METHOD:  $SOLVE"
    for (( LEV=3; LEV<=$MAXLEV; LEV++ )); do
        rm -rf tmp.txt
        ../obstacle $SOLVE -snes_grid_sequence $LEV -log_view > tmp.txt
        grep "errors:" tmp.txt
        grep "last KSP iters" tmp.txt
        grep "Flop:  " tmp.txt
    done
    echo
done

