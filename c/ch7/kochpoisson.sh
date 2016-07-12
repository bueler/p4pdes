#!/bin/bash

# the following "fun" example:
#    generate Koch snowflake fractal in polygon file meshes/koch.poly
#    triangulate it (see meshes/koch.1.*)
#    solve Poisson equation on it (-grad^2 u = 1 with u=0 Dirichlet)
#    generate a figure "snowflake.pdf" showing u(x,y) as a contour map

make unfem

cd meshes
../kochmesh.py -l 5
triangle -pqa0.00002 koch.poly
#showme koch
../tri2petsc.py koch.1 koch.1
../unfem -un_mesh koch.1 -un_case 4 -un_view_solution -snes_monitor -ksp_converged_reason
../petsc2contour.py -i koch.1 -o ../snowflake.pdf --contours 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.02 0.03 0.05 0.1

