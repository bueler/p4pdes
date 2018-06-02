#!/usr/bin/env python
#
# (C) 2016-2018 Ed Bueler

from __future__ import print_function
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate a .geo file for a Koch snowflake.')
parser.add_argument('--neumann', action='store_true', default=False,
                    help='mark entire boundary as Neumann (useful for plotting polygon only; see petsc2tikz.py)')
parser.add_argument('-l', type=int, metavar='LEVEL', default=2,
                    help='recursion level for snowflake; default=2')
parser.add_argument('-o', default='koch.geo', metavar='FILE',
                    help='name for .geo output file; default=koch.geo')
args = parser.parse_args()
lev = args.l

N = 3 * 4**lev   # number of boundary nodes
print('creating level %d Koch snowflake polygon with %d segments ...' % (lev,N))

# generate string code for snowflake; see https://commons.wikimedia.org/wiki/Koch_snowflake
koch_flake = "FRFRF"
for i in range(lev):
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

# get length of a first segment
h0 = np.sqrt(sum((nodes[1,:] - nodes[0,:])**2))

# generate Gmsh-readable .geo file
print('writing Gmsh-readable file  %s  ...' % args.o)
f = open(args.o, 'w')
f.write('// level %d Koch domain with %d boundary segments\n\n' % (lev,N))
f.write('cl = %.10e; // characteristic length\n' % h0)
for j in range(N):
    f.write('Point(%d) = {%.10f,%.10f,0,cl};\n' % (j+1,nodes[j,0],nodes[j,1]))
for j in range(N-1):
    f.write('Line(%d) = {%d,%d};\n' % (N+1+j,j+1,j+2))
f.write('Line(%d) = {%d,1};\n' % (2*N,N))
f.write('Line Loop(%d) = {' % (2*N+1))
for j in range(N-1):
    f.write('%d,' % (N+1+j))
f.write('%d};\n' % (2*N))
f.write('Plane Surface(%d) = {%d};\n' % (2*N+2,2*N+1))
if args.neumann:
    f.write('Physical Line("neumann") = {')
else:
    f.write('Physical Line("dirichlet") = {')
for j in range(N-1):
    f.write('%d,' % (N+1+j))
f.write('%d};\n' % (2*N))
if args.neumann:
    f.write('Physical Line("dirichlet") = {};\n')
else:
    f.write('Physical Line("neumann") = {};\n')
f.write('Physical Surface("interior") = {%d};\n' % (2*N+2))
f.close()

