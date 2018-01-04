#!/bin/bash

# generate levels of Koch polygons, with re-triangulation at each level,
# using triangle and tri2petsc.py

# run "make test" first to get links to PETSc python scripts for binary files

# example:
#   ./genkochs.sh koch 7
#   showme koch

NAME=$1
MAXLEV=$2

area[1]=0.1
area[2]=0.025
area[3]=0.005
area[4]=0.001
area[5]=0.00025
area[6]=0.00005
area[7]=0.00001
area[8]=0.0000025
area[9]=0.0000005

for (( Z=1; Z<=$MAXLEV; Z++ )); do
    # generate .poly for level Z Koch polygon
    rm -f tmp.poly
    ./polygon.py -l $Z -o tmp.poly
    # generates .poly, .node, .ele
    triangle -pqa${area[$Z]} tmp.poly
    mv tmp.1.poly $NAME.$Z.poly
    mv tmp.1.node $NAME.$Z.node
    mv tmp.1.ele $NAME.$Z.ele
    # generates .vec, .is
    ../tri2petsc.py $NAME.$Z $NAME.$Z
done
rm -f tmp.poly

