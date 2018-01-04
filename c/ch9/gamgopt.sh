#!/bin/bash

# test -pc_type gamg for optimality by running unfem on a sequence of refining
# meshes; uses CG for Krylov

# run gensquares.sh, gentraps.sh, or koch/genkochs.sh first

# example:
#   ./gamgopt.sh meshes/sq 3 4      # -un_case 3 and 4 levels
#   ./gamgopt.sh meshes/sq 3 6      # ... level 6 is large: 3200x3200 grid w. N=10^7
#   ./gamgopt.sh meshes/trap 0 6    # -un_case 0 and 6 levels
#   ./gamgopt.sh koch/koch 4 7      # -un_case 4 and 7 levels
#   ./gamgopt.sh koch/koch 4 9      # ... level 9 is large: N = 4 x 10^6

# run as:
#   ./gamgopt.sh koch/koch 4 9 &> gamgopt.txt
# use PETSC_ARCH with --with-debugging=0

# results & figure-generation:  see p4pdes-book/figs/gamgopt.txt|py

NAME=$1
CASE=$2
MAXLEV=$3

for (( Z=1; Z<=$MAXLEV; Z++ )); do
    echo "running level ${Z}"
    IN=$NAME.$Z
    cmd="./unfem -un_case $CASE -snes_type ksponly -ksp_rtol 1.0e-8 -ksp_type cg -pc_type gamg -snes_monitor_short -ksp_converged_reason -un_mesh $IN"
    rm -f foo.txt
    $cmd -log_view &> foo.txt
    'grep' "Linear solve converged due to" foo.txt
    'grep' "result for N" foo.txt
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
done
rm -f foo.txt

