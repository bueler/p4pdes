#!/usr/bin/env python
#
# (C) 2014-2018 Ed Bueler

from __future__ import print_function

import sys, argparse
import numpy as np
import PetscBinaryIO  # may use link

petsc = PetscBinaryIO.PetscBinaryIO()

parser = argparse.ArgumentParser(description='Generate a structured grid on the unit square in PETSc binary format (.vec,.is), readable by ch9/unfem.')
# positional args; both required
parser.add_argument('root', metavar='NAMEROOT',
                    help='output file name root')
parser.add_argument('M', type=int,
                    help='mesh has N = MxM points')
parser.add_argument('-debug', default=False, action='store_true',
                    help='print vectors when generated')
args = parser.parse_args()
vecname = args.root + '.vec'
isname = args.root + '.is'
N = args.M * args.M
h = 1.0 / ((float)(args.M) - 1.0)

def n(i,j):                 # computes node index from local
    return j*args.M + i

# write node locations to .vec file
print('  creating N=%d node locations ...' % N)
xy = np.zeros(2*N)
for j in range(args.M):
    for i in range(args.M):
        xy[2*n(i,j):2*(n(i,j)+1)] = [(float)(i) * h, (float)(j) * h]
if args.debug:
  print(xy)
print('  writing node locations as PETSc Vec to %s ...' % vecname)
petsc.writeBinaryFile(vecname,[xy.view(PetscBinaryIO.Vec),])

# create element triples
K = (args.M-1)*(args.M-1) * 2   # two triangles per square cell
print('  creating K=%d element triples ...' % K)
e = np.zeros(3*K,dtype=int)
for j in range(args.M-1):
    for i in range(args.M-1):
        # write first triangle in cell
        k = 2*(j*(args.M-1) + i)
        A = n(i,j)
        B = A + 1
        C = n(i,j+1)
        e[3*k:3*(k+1)] = [A,B,C]
        # write second triangle in cell
        k += 1
        A1 = B
        B1 = C + 1
        C1 = C
        e[3*k:3*(k+1)] = [A1,B1,C1]
if args.debug:
  print(e)

# create boundary flags
print('  creating N=%d boundary flags ...' % N)
bf = np.zeros(N,dtype=int)
for j in range(args.M):
    for i in range(args.M):
        if (i == 0) or (j == 0) or (i == args.M-1) or (j == args.M-1):
            bf[n(i,j)] = 2
if args.debug:
  print(bf)

# create bogus negative Neumann boundary segment
# FIXME this kluge caused by inability to check if binary file is empty; see FIXME in UMReadISs()
ns = np.array([-1,-1],dtype=int)

# write ISs
print('  writing element triple and boundary flags as PETSc IS to %s ...' % isname)
IS = PetscBinaryIO.IS
petsc.writeBinaryFile(isname,[e.view(IS),bf.view(IS),ns.view(IS)])

