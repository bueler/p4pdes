koch/
=====

This directory contains script `poisson.sh` which does the following for-fun
example:

1. Generate a Koch snowflake polygon, a pseudo-fractal, in triangle-readable
file `koch.poly`.
2. Triangulates the interior S of the polygon and generates `unfem`-readable
files `koch.1.*`.
3. Solves the Poisson equation,  -grad^2 u = 1,  with u=0 Dirichlet boundary
conditions, on S.
4. Generates an image `snowflake.pdf` showing u(x,y) as a contour map.

Before running the example do

        $ (cd ../ && make unfem test)

To run the example:

        $ ./poisson.sh

Now use a PDF viewer on `snowflake.pdf`.

To clean up the stuff in this directory, do

        $ make clean

