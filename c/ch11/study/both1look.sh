#!/bin/bash
set -e

# just show both1 solutions on coarse grid

# this script generates both1look_{none,centered,vanleer}.txt which are
# loaded by figure script p4pdes-book/figs/both1look.py

for LIM in none centered vanleer; do
    ../both1 -da_refine 3 -snes_fd_color -snes_converged_reason -ksp_converged_reason \
        -snes_view_solution ascii:$LIM.m:ascii_matlab -b1_limiter $LIM
    # remove first three and last one lines
    tail -n +4 $LIM.m | head -n -1 > both1look_$LIM.txt
    rm $LIM.m
done
