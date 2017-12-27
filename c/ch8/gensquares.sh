#!/bin/bash

# generate square structured meshes using genstructured.py and tri2petsc.py
# see also gamgopt.sh for unfem runs using these

# run "make test" first to get links to PETSc python scripts for binary files

# example:
#   ./gensquares.sh sq 4
#   ./gensquares.sh sq 6    # level 6 is large: 3200x3200 grid

NAME=$1
LEV=$2

MX=100
for (( N=1; N<=$LEV; N++ )); do
    echo "generating level $N square mesh with MX=$MX"
    OUT=meshes/$NAME.$N
    ./genstructured.py $OUT $MX
    ./tri2petsc.py $OUT
    MX=$((MX*2))
done

