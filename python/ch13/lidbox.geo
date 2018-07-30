// box domain geometry for lid-driven cavity example
// usage to generate lidbox.msh for input in stokes.py:
//   $ gmsh -2 lidbox.geo

cl = 0.1;       // characteristic length: for majority of domain
cleddy = 0.005; //                        for Moffatt eddy corners
trans = 0.4;    // location of transition to refined eddy corners

Point(1) = {0.0,1.0,0,cl};
Point(2) = {0.0,trans,0,cl};
Point(3) = {0.0,0.0,0,cleddy};
Point(4) = {trans,0.0,0,cl};
Point(5) = {1.0-trans,0.0,0,cl};
Point(6) = {1.0,0.0,0,cleddy};
Point(7) = {1.0,trans,0,cl};
Point(8) = {1.0,1.0,0,cl};

Line(10) = {1,2};
Line(11) = {2,3};
Line(12) = {3,4};
Line(13) = {4,5};
Line(14) = {5,6};
Line(15) = {6,7};
Line(16) = {7,8};
Line(17) = {8,1};

Line Loop(20) = {10,11,12,13,14,15,16,17};
Plane Surface(30) = {20};

Physical Line(40) = {17};  // lid
Physical Line(41) = {10,11,12,13,14,15,16};  // other

Physical Surface(50) = {30};  // interior

