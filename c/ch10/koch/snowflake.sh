#!/bin/bash

LEVEL=$1

echo "**** generating mesh ****"
# generate and triangulate a Koch polygon
./domain.py -l $LEVEL -o koch.geo
gmsh -2 koch.geo
../msh2petsc.py koch.msh

echo
echo "**** solving Poisson equation ****"
# solve the Poisson equation  -grad^2 u = 2  on the Koch domain using AMG
CMD="../unfem -un_mesh koch -un_case 4 -snes_type ksponly -ksp_rtol 1.0e-8 -pc_type gamg -un_view_solution -ksp_monitor -ksp_converged_reason"
echo "command = $CMD"
$CMD

echo
echo "**** generating figure ****"
# report  max u(x,y) = u(0,0)  and generate contour plot
CONTOURS="1e-6 1e-5 3e-5 1e-4 3e-4 0.001 0.003 0.01 0.02 0.03 0.05 0.1 0.2"
../vis/petsc2contour.py -i koch -o snowflake.pdf --contours $CONTOURS

