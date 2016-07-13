#!/bin/bash

LEVEL=5
ELEMENTAREA=0.00002
CONTOURS="1e-6 1e-5 3e-5 1e-4 3e-4 0.001 0.003 0.01 0.02 0.03 0.05 0.1"

# triangulate the Koch snowflake
./mesh.py -l $LEVEL
triangle -pqa$ELEMENTAREA koch.poly
#showme koch

# solve the Poisson equation on it
../tri2petsc.py koch.1 koch.1
../unfem -un_mesh koch.1 -un_case 4 -un_view_solution \
    -snes_monitor -ksp_converged_reason

# generate contour plot
../petsc2contour.py -i koch.1 -o snowflake.pdf --contours $CONTOURS

