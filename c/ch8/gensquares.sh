#!/bin/bash

# run "make test" first
# see also gamgopt.sh for unfem runs

#FIXME consider names  meshes/sq$LEV.1.xxx so that names are like koch$LEV.1.xxx
#FIXME option for MAXLEV
MX=100
for LEV in 1 2 3 4 5 6; do
    echo "generating level $LEV square mesh with MX=$MX"
    ./genstructured.py meshes/sq$LEV $MX
    ./tri2petsc.py meshes/sq$LEV meshes/sq$LEV
    MX=$((MX*2))
done

