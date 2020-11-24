p4pdes/c/ch10/
==============

This directory contains programs for Chapter 10.  Most runs will need Gmsh

### software testing

Software testing is the usual:

    $ make test

Note that the first test is of the Gmsh version.  If this test fails then
it merely means your version is different from the one used to save the test
output.  Usually everything else is actually fine.  However, the details of
meshes may change between Gmsh versions.  This can cause some of the other
tests to fail because the number of mesh vertices etc., or number of
iterations, can change.

