#!/bin/bash

cd meshes/
triangle -pqa0.5 trap
triangle -rpqa0.1 trap.1
triangle -rpqa0.02 trap.2
triangle -rpqa0.005 trap.3
triangle -rpqa0.001 trap.4
triangle -rpqa0.0002 trap.5
cd ..

#LEVS="1 2 3 4 5 6"
LEVS="1 2 3 4"

for N in $LEVS; do
    ./tri2petsc.py meshes/trap.$N trap.$N.dat
done

make unfem
for N in $LEVS; do
    rm -f foo.txt
    ./unfem -un_mesh trap.$N.dat -snes_fd -quaddeg 1 -snes_max_funcs 100000 -log_view &> foo.txt
    'grep' result foo.txt
    'grep' SNESFunctionEval foo.txt
done
rm -f foo.txt

