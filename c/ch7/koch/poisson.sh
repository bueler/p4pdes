#!/bin/bash

./mesh.py -l 5
triangle -pqa0.00002 koch.poly
#showme koch
../tri2petsc.py koch.1 koch.1
../unfem -un_mesh koch.1 -un_case 4 -un_view_solution \
    -snes_monitor -ksp_converged_reason
../petsc2contour.py -i koch.1 -o snowflake.pdf \
    --contours 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.02 0.03 0.05 0.1

