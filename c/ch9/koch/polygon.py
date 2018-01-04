#!/usr/bin/env python
#
# (C) 2016 Ed Bueler

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate a .poly file for a Koch snowflake.')
parser.add_argument('-l', type=int, metavar='LEVEL', default=3, help='recursion level for snowflake; default=3; note levels 1 .. 8 finish in seconds')
parser.add_argument('-o', default='koch.poly', metavar='FILE', help='output filename; default=koch.poly')
args = parser.parse_args()

iterations = args.l

N = 3 * 4**iterations   # number of nodes
print "creating level %d Koch snowflake polygon with P=%d segments ..." % (iterations,N)

# generate string code for snowflake; see https://commons.wikimedia.org/wiki/Koch_snowflake
koch_flake = "FRFRF"
for i in range(iterations):
    koch_flake = koch_flake.replace("F","FLFRFLF")

# draw by saving array of "turtle" positions
nodes = np.zeros((N,2))
pos = np.array([0.0,0.0])  # initial location; arbitrary because of rescale (below)
vec = np.array([1.0,0.0])  # initial direction
dist = 1.0  # edge length; really arbitrary because of rescale
count = 0
for move in koch_flake:
    if move == "F":
        nodes[count] = pos
        count += 1
        pos += dist * vec
    elif move == "L":
        cl = np.cos(np.pi/3.0)
        sl = np.sin(np.pi/3.0)
        vec = np.array([cl * vec[0] - sl * vec[1], sl * vec[0] + cl * vec[1]])
    elif move == "R":
        cl = np.cos(-2.0*np.pi/3.0)
        sl = np.sin(-2.0*np.pi/3.0)
        vec = np.array([cl * vec[0] - sl * vec[1], sl * vec[0] + cl * vec[1]])

# rescale so it is centered at (0,0) and fits into unit circle
c = np.average(nodes,axis=0)
for j in range(N):
    nodes[j] -= c
r = np.sqrt(nodes[:,0]**2+nodes[:,1]**2).max()
for j in range(N):
    nodes[j] /= r

# generate triangle-readable .poly file
print "writing triangle-readable file  %s  ..." % args.o
f = open(args.o, 'w')
f.write('# N=%d nodes (vertices), in 2D, no attributes, no vertex markers\n' % N)
f.write('%d 2 0 0\n' % N)
for j in range(N):
    f.write('%5d %15.10f %15.10f\n' % (j,nodes[j,0],nodes[j,1]))
f.write('# P=%d segments (edges), one edge marker (2 = Dirichlet)\n' % N)
f.write('%d 1\n' % N)
for j in range(N-1):
    f.write('%5d %4d %4d   2\n' % (j,j,j+1))
f.write('%5d %4d    0   2\n' % (N-1,N-1))
f.write('# zero holes; it is simply-connected\n0\n')
f.close()

