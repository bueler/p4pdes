#!/bin/bash
set -e
set +x

# run with --with-debugging=0 build, as
#   ./obstaclemgruns.sh &> obstaclemgruns.txt

# generates first two tables in Chapter 12

# see also p4pdes-book/figs/obstacle.py for visualization

# defaults to RS solver

MAXLEV=7   # to 257x257 grid

for GRID in "-da_refine" "-snes_grid_sequence"; do
    echo "=============== USING $GRID ============="
    for SNES in vinewtonrsls vinewtonssls; do
        for PCMG in "-pc_type mg -mg_levels_ksp_max_it 3" \
                    "-pc_type gamg -pc_gamg_type classical"; do
            METH="-snes_type $SNES -ksp_type cg $PCMG"
            echo "METHOD:  $METH"
            for (( LEV=3; LEV<=$MAXLEV; LEV++ )); do
                rm -rf tmp.txt
                ../obstacle $GRID $LEV $METH -log_view > tmp.txt
                grep "errors:" tmp.txt
                grep "last KSP iters" tmp.txt
                grep "Flop:  " tmp.txt
            done
            echo
        done
    done
done

