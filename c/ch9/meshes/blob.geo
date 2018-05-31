// A polygonal domain.  Look at it with:
//     gmsh blob.geo
// To generate a initial coarse mesh and refine it twice do:
//     gmsh -2 blob.geo -o blob1.msh
//     gmsh -refine blob1.msh -o blob2.msh
// Look at the results:
//     gmsh blobX.msh

cl = 4.0;  // characteristic length
Point(1) = {0.0,0.0,0,cl};
Point(2) = {0.0,2.0,0,cl};
Point(3) = {5.0,2.0,0,cl};
Point(4) = {8.0,0.0,0,cl};
Point(5) = {6.0,-3.0,0,cl};
Point(6) = {1.0,-5.0,0,cl};
Point(7) = {-1.0,-4.0,0,cl};
Point(8) = {2.0,-3.0,0,cl};
Point(9) = {0.0,-2.0,0,cl};
Line(10) = {1,2};
Line(11) = {2,3};
Line(12) = {3,4};
Line(13) = {4,5};
Line(14) = {5,6};
Line(15) = {6,7};
Line(16) = {7,8};
Line(17) = {8,9};
Line(18) = {9,1};
Line Loop(19) = {10,11,12,13,14,15,16,17,18};
Plane Surface(20) = {19};
Physical Line("dirichlet") = {10,11,12};
Physical Line("neumann") = {13,14,15,16,17,18};
Physical Surface("interior") = {20};

