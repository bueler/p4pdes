#!/bin/bash

LEVEL=5

# triangulate a Koch polygon
./domain.py -l $LEVEL -o koch.geo
gmsh -2 koch.geo

# solve the Poisson equation on it
../msh2petsc.py koch.msh
../unfem -un_mesh koch -un_case 4 -snes_type ksponly -pc_type gamg -un_view_solution -snes_monitor -ksp_converged_reason

# generate contour plot
CONTOURS="1e-6 1e-5 3e-5 1e-4 3e-4 0.001 0.003 0.01 0.02 0.03 0.05 0.1"
../vis/petsc2contour.py -i koch -o snowflake.pdf --contours $CONTOURS

