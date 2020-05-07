#!/bin/bash
set -e
set +x

# generates data read by p4pdes-book/figs/poissoneigs.py to generate a figure
# in chapter 3

POIS="../poisson"

$POIS -da_grid_x 5 -da_grid_y 5 -mat_view binary:poissonmat5.dat
$POIS -da_refine 0 -mat_view binary:poissonmat9.dat
$POIS -da_refine 1 -mat_view binary:poissonmat17.dat
$POIS -da_refine 2 -mat_view binary:poissonmat33.dat
$POIS -da_refine 3 -mat_view binary:poissonmat65.dat
$POIS -da_refine 4 -mat_view binary:poissonmat129.dat

