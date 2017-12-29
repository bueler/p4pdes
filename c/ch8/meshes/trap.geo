/*
trap.geo describes same polygon (geometry) as trap.poly, but suitable for
reading by gmsh; do
    $ gmsh trap.geo
and then generate a 2D mesh trap.msh in the gui, or do
    $ gmsh -2 trap.geo
*/

c = 0.4;   // characteristic length  FIXME what is best? can we avoid choosing?
Point(1) = { 2.0, 0.0, 0, c};
Point(2) = { 1.0, 1.0, 0, c};
Point(3) = {-1.0, 1.0, 0, c};
Point(4) = {-2.0, 0.0, 0, c};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// FIXME boundary markers for points and lines
