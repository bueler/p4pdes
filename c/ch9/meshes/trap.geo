// trapezoid domain geometry
// usage:  gmsh -2 trap.geo

cl = 1.5;  // characteristic length
Point(1) = {2.0,0.0,0,cl};
Point(2) = {1.0,1.0,0,cl};
Point(3) = {-1.0,1.0,0,cl};
Point(4) = {-2.0,0.0,0,cl};
Line(5) = {1,2};
Line(6) = {2,3};
Line(7) = {3,4};
Line(8) = {4,1};
Line Loop(9) = {5,6,7,8};
Plane Surface(10) = {9};
Physical Line("dirichlet") = {5,6,7};
Physical Line("neumann") = {8};
Physical Surface("interior") = {10};

