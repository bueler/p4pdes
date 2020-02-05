#!/bin/bash

# test -pc_type gamg for optimality by running unfem on a sequence of refining
# meshes; uses CG for Krylov

# generate meshes/trapN.{is,vec} for N=1,...,12 first

# main example uses PETSC_ARCH with --with-debugging=0:
#   cd c/ch10/
#   ./refinetraps.sh meshes/trap 12
#   cd study/
#   ./gamgopt.sh ../meshes/trap 0 12 &> gamgopt.txt

# LEV=12 with N=10^7 nodes and ~100 sec run time achievable given ~10 Gb memory

# results & figure-generation:  see p4pdes-book/figs/gamgopt.txt|py

NAME=$1
CASE=$2
MAXLEV=$3

for (( Z=1; Z<=$MAXLEV; Z++ )); do
    echo "running level ${Z}"
    IN=${NAME}$Z
    cmd="../unfem -un_case $CASE -snes_type ksponly -pc_type gamg -ksp_rtol 1.0e-10 -ksp_converged_reason -un_mesh $IN"
    rm -f foo.txt
    echo $cmd
    $cmd -log_view &> foo.txt
    'grep' "Linear solve converged due to" foo.txt
    'grep' "result for N" foo.txt
    'grep' "Flop:  " foo.txt | awk '{print $2}'
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
done
rm -f foo.txt

