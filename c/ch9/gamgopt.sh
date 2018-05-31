#!/bin/bash

# test -pc_type gamg for optimality by running unfem on a sequence of refining
# meshes; uses CG for Krylov

# main example uses PETSC_ARCH with --with-debugging=0:
#   ./refinetraps.sh meshes/trap LEV
#   ./gamgopt.sh meshes/trap 0 LEV &> gamgopt.txt
# LEV=12 with N=10492929 nodes and 100 sec run achievable on ed-galago

# other examples: FIXME redo with genstructured.py and koch/ stuff
#   ./gamgopt.sh meshes/sq 3 6      # ... level 6 is large: 3200x3200 grid w. N=10^7
#   ./gamgopt.sh koch/koch 4 9      # ... level 9 is large: N = 4 x 10^6

# results & figure-generation:  see p4pdes-book/figs/gamgopt.txt|py

NAME=$1
CASE=$2
MAXLEV=$3

for (( Z=1; Z<=$MAXLEV; Z++ )); do
    echo "running level ${Z}"
    IN=${NAME}$Z
    cmd="./unfem -un_case $CASE -snes_type ksponly -ksp_rtol 1.0e-9 -ksp_type cg -pc_type gamg -ksp_converged_reason -un_mesh $IN"
    rm -f foo.txt
    echo $cmd
    $cmd -log_view &> foo.txt
    'grep' "Linear solve converged due to" foo.txt
    'grep' "result for N" foo.txt
    'grep' "Flop:  " foo.txt | awk '{print $2}'
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
done
rm -f foo.txt

