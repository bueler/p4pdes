#!/bin/bash

# generate refining trapezoidal unstructured meshes using Gmsh and msh2petsc.py
# refinement is by splitting elements using "gmsh -refine"

# run "make test" first to get links to PETSc python scripts for binary files

# example:
#   ./gentraps.sh trap 5

NAME=$1
MAXLEV=$2

gmsh -2 meshes/${NAME}.geo -o meshes/${NAME}1.msh
./msh2petsc.py meshes/${NAME}1.msh
for (( Z=1; Z<$MAXLEV; Z++ )); do
    gmsh -refine meshes/${NAME}$Z.msh -o meshes/${NAME}$((Z+1)).msh
    ./msh2petsc.py meshes/$NAME$((Z+1)).msh
done

