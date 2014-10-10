#!/usr/bin/env python
#
# (C) 2014 Ed Bueler


import numpy
import argparse
import sys

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='Generate a structured grid on the unit square in triangle format (three files: .node, .poly, .ele).')
# positional args; both required
parser.add_argument('file', metavar='NAMEROOT',
                    help='root of output file name for .node,.ele,.poly')
parser.add_argument('N',
                    help='mesh has NxN points')

# process options: simpler names, and type conversions
args = parser.parse_args()
N    = (int)(args.N)
h    = 1.0 / ((float)(N) - 1.0)

# write .node file
node = open(args.file + '.node', 'w')
print 'writing file %s with %d nodes ...' % (args.file + '.node',N*N)
node.write('%d  2  0  1\n' % (N*N))  # N^2 nodes, 2D, no attributes, one bdry marker
for j in range(N):
    for i in range(N):
        n = j*N + i + 1
        x = (float)(i) * h
        y = (float)(j) * h
        if (i == 0) or (i == N-1) or (j == 0) or (j == N-1):
            bdry = 2  # only Dirichlet boundary
        else:
            bdry = 0
        node.write(' %4d  %g  %g  %4d\n' % (n, x, y, bdry))
node.write('# created by: %s\n' % commandline)
node.close()

# write .ele file
ele = open(args.file + '.ele', 'w')
K = (N-1)*(N-1) * 2   # two triangles per square cell
print 'writing file %s with %d eles ...' % (args.file + '.ele',K)
ele.write('%d  3  0\n' % K)  # K triangles, 3 nodes per triangle, no attributes
for j in range(N-1):
    for i in range(N-1):
        # write two triangles
        k = 2*(j*(N-1) + i) + 1
        A = j*N + i + 1
        B = A+1
        C = (j+1)*N + i + 1
        ele.write(' %4d  %4d  %4d  %4d\n' % (k, A, B, C))
        k += 1
        A1 = C
        B1 = C+1
        C1 = B
        ele.write(' %4d  %4d  %4d  %4d\n' % (k, A1, B1, C1))
ele.write('# created by: %s\n' % commandline)
ele.close()

# write .poly file
poly = open(args.file + '.poly', 'w')
print 'writing file %s ...' % (args.file + '.poly')
bottomleft  = 1
bottomright = N
topright    = N*N
topleft     = N*(N-1)+1
poly.write('0  2  0  1\n')        # no nodes in this file
poly.write('4  1\n')              # four boundary segments, one marker
poly.write(' 1  %4d  %4d    2\n' % (bottomleft ,bottomright)) # sides cycle; Dirichlet = 2
poly.write(' 2  %4d  %4d    2\n' % (bottomright,topright))
poly.write(' 3  %4d  %4d    2\n' % (topright,   topleft))
poly.write(' 4  %4d  %4d    2\n' % (topleft,    bottomleft))
poly.write('0\n')                 # no holes
poly.write('# created by: %s\n' % commandline)
poly.close()




