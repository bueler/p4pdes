koch/
=====

This directory contains script `snowflake.sh` which does the following example:

1. Generate a Koch snowflake polygon in Gmsh-readable `.geo` domain format.
2. Triangulate the polygon using Gmsh, giving mesh file `koch.msh`.
3. Generate PETSc binary files `koch.vec`, `koch.is` for the mesh.
4. Use `unfem.c` to solve the Poisson equation -grad^2 u = 2, with u=0 Dirichlet boundary conditions.
5. Report  max u(x,y) = u(0,0)  and generate an image `snowflake.pdf` showing u(x,y) as a contour map.

Before running the example do

        $ (cd ../ && make test)

To run the example:

        $ ./snowflake.sh 5

Now use a PDF viewer on `snowflake.pdf`.

To clean up the stuff in this directory, do

        $ make clean

