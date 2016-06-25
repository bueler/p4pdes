#!/bin/bash

# generate trapezoidal unstructured meshes using triangle and tri2petsc.py

# example:
#   ./gentraps.sh trap 5

NAME=$1
LEV=$2

area[0]=0.5
area[1]=0.1
area[2]=0.02
area[3]=0.005
area[4]=0.001
area[5]=0.0002
area[6]=0.00005
area[7]=0.00001
area[8]=0.000002
triangle -pqa${area[0]} meshes/$NAME
for (( N=1; N<$LEV; N++ )); do
    # generate .poly, .node, .ele for meshes
    triangle -rpqa${area[$N]} meshes/$NAME.$N
    # generate .vec, .is files
    ./tri2petsc.py meshes/$NAME.$N meshes/$NAME.$N
done

