koch/
=====

This directory contains script `snowflake.sh` which does a for-fun example:

1. Generate a Koch snowflake polygon in triangle-readable `.poly` file.
2. Triangulates the polygon and generates `unfem`-readable files `koch.1.*`.
3. Solves the Poisson equation,  -grad^2 u = 1,  with u=0 Dirichlet boundary
conditions.
4. Generates an image `snowflake.pdf` showing u(x,y) as a contour map.

Before running the example do

        $ (cd ../ && make unfem test)

To run the example:

        $ ./snowflake.sh

Now use a PDF viewer on `snowflake.pdf`.

To clean up the stuff in this directory, do

        $ make clean

