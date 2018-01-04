#!/usr/bin/env python
#
# (C) 2014-2016 Ed Bueler


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

def n(i,j):                 # computes node index from local
    return j*N + i

# write .node file
node = open(args.file + '.node', 'w')
print 'writing file %s with %d nodes ...' % (args.file + '.node',N*N)
node.write('%d  2  0  1\n' % (N*N))  # N^2 nodes, 2D, no attributes, one bdry marker
for j in range(N):
    for i in range(N):
        x = (float)(i) * h
        y = (float)(j) * h
        if (i == 0) or (i == N-1) or (j == 0) or (j == N-1):
            bdry = 2  # only Dirichlet boundary
        else:
            bdry = 0
        node.write(' %4d  %.14f  %.14f  %4d\n' % (n(i,j), x, y, bdry))
node.write('# created by command: %s\n' % commandline)
node.close()

# write .ele file
ele = open(args.file + '.ele', 'w')
K = (N-1)*(N-1) * 2   # two triangles per square cell
print 'writing file %s with %d elements ...' % (args.file + '.ele',K)
ele.write('%d  3  0\n' % K)  # K triangles, 3 nodes per triangle, no attributes
for j in range(N-1):
    for i in range(N-1):
        # write first triangle in cell
        k = 2*(j*(N-1) + i)
        A = n(i,j)
        B = A + 1
        C = n(i,j+1)
        # write second triangle in cell
        ele.write(' %4d  %4d  %4d  %4d\n' % (k, A, B, C))
        k += 1
        A1 = B
        B1 = C + 1
        C1 = C
        ele.write(' %4d  %4d  %4d  %4d\n' % (k, A1, B1, C1))
ele.write('# created by: %s\n' % commandline)
ele.close()

# write .poly file
poly = open(args.file + '.poly', 'w')
M = 4*(N-1)          # N-1 boundary segments on each side
print 'writing file %s with %d boundary segments ...' % (args.file + '.poly',M)
bottomleft  = 1
bottomright = N
topright    = N*N
topleft     = N*(N-1)+1
poly.write('0  2  0  1\n')          # no nodes in this file
poly.write('%d  1\n' % (M))         # M boundary segments, one marker
count = 0
j = 0                               # bottom segments
for i in range(N-1):
    poly.write(' %4d  %4d  %4d    2\n' % (count,n(i,j),n(i,j)+1))
    count += 1
i = N-1                             # right segments
for j in range(N-1):
    poly.write(' %4d  %4d  %4d    2\n' % (count,n(i,j),n(i,j)+N))
    count += 1
j = N-1                             # top segments
for i in range(N-1,0,-1): # count down
    poly.write(' %4d  %4d  %4d    2\n' % (count,n(i,j),n(i,j)-1))
    count += 1
i = 0                               # left segments
for j in range(N-1,0,-1): # count down
    poly.write(' %4d  %4d  %4d    2\n' % (count,n(i,j),n(i,j)-N))
    count += 1
poly.write('0\n')                 # no holes
poly.write('# created by: %s\n' % commandline)
poly.close()

