#!/bin/bash

# generate refining trapezoidal unstructured meshes using triangle and tri2petsc.py

# run "make test" first to get links to PETSc python scripts for binary files

# example:
#   ./gentraps.sh trap 5

NAME=$1
MAXLEV=$2

area[0]=0.5
area[1]=0.1
area[2]=0.02
area[3]=0.005
area[4]=0.001
area[5]=0.0002
area[6]=0.00005
area[7]=0.00001
area[8]=0.000002
area[9]=0.0000005
triangle -pqa${area[0]} meshes/$NAME
./tri2petsc.py meshes/$NAME.1
for (( Z=1; Z<$MAXLEV; Z++ )); do
    # generates .poly, .node, .ele
    triangle -rpqa${area[$Z]} meshes/$NAME.$Z
    # generates .vec, .is
    OUT=meshes/$NAME.$((Z+1))
    ./tri2petsc.py $OUT
done

