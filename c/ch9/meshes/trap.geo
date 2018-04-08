// generate trapezoid domain geometry
// usage:  gmsh -2 trap.geo

lc = 1.5;
Point(1) = {2.0,0.0,0,lc};
Point(2) = {1.0,1.0,0,lc};
Point(3) = {-1.0,1.0,0,lc};
Point(4) = {-2.0,0.0,0,lc};
Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};
Line Loop(9) = {5,6,7,8};
Plane Surface(10) = {9};

