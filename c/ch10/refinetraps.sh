#!/bin/bash

# generate refining trapezoidal unstructured meshes using Gmsh and msh2petsc.py
# refinement is by splitting elements using "gmsh -refine"

# run "make test" first to get links to PETSc python scripts for binary files

# example:
#   ./refinetraps.sh meshes/trap 5

NAME=$1
MAXLEV=$2

GMSH=gmsh
#GMSH="gmsh -format msh22"   # legacy 2.2 format also allowed with msh2petsc.py

${GMSH} -2 ${NAME}.geo -o ${NAME}1.msh
./msh2petsc.py ${NAME}1.msh
for (( Z=1; Z<$MAXLEV; Z++ )); do
    ${GMSH} -refine ${NAME}$Z.msh -o ${NAME}$((Z+1)).msh
    ./msh2petsc.py $NAME$((Z+1)).msh
done

