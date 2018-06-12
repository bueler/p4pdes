#!/usr/bin/env python
#
# (C) 2016-2018 Ed Bueler

from __future__ import print_function
import sys, argparse
import numpy as np

parser = argparse.ArgumentParser(description=
'''Contour plot of a solution on a triangulation.  Reads PETSc binary
files .vec,.is,.soln.  (Note .soln file is generated from unfem option
-un_view_solution.)  Requires ../PetscBinaryIO.py.''')
parser.add_argument('-i', metavar='ROOT',
                    help='root of input file name for files .vec,.is,.soln')
parser.add_argument('-o', metavar='PDFFILE', default='',
                    help='output file name (image in PDF format)')
parser.add_argument('--contours', metavar='C', type=np.double, nargs='+',
                    default=np.nan, help='contour levels; space-separated list')
args = parser.parse_args()

import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.append('../')
import PetscBinaryIO # may need link

root = args.i

io = PetscBinaryIO.PetscBinaryIO()
vecfile  = open(root+'.vec')
isfile   = open(root+'.is')
solnfile = open(root+'.soln')

print('reading triangulation from files %s,%s ...' % (root+'.vec',root+'.is'))
objecttype = io.readObjectType(vecfile)
if objecttype == 'Vec':
    xy = io.readVec(vecfile)
    N = len(xy)
    if N % 2 != 0:
        print('ERROR: nodes in .vec file invalid ... stopping')
        sys.exit()
    N /= 2
    xy = np.reshape(xy,(N,2))
else:
    print('ERROR: no valid .vec file ... stopping')
    sys.exit()

objecttype = io.readObjectType(isfile)
if objecttype == 'IS':
    ele = io.readIS(isfile)
    K = len(ele)
    if K % 3 != 0:
        print('ERROR: elements (triangles) in .is file invalid ... stopping')
        sys.exit()
    K /= 3
    if (ele.max() > N) or (ele.min() < 0):
        print('ERROR: elements contain invalid indices ... stopping')
        sys.exit()
    ele = np.reshape(ele,(K,3))
else:
    print('ERROR: no valid .is file ... stopping')
    sys.exit()

print('reading solution from file %s ...' % (root+'.soln'))
objecttype = io.readObjectType(solnfile)
if objecttype == 'Vec':
    u = io.readVec(solnfile)
    if len(u) != N:
        print('ERROR: solution vec is wrong size ... stopping')
        sys.exit()
else:
    print('ERROR: no valid .soln file ... stopping')
    sys.exit()

x = xy[:, 0]
y = xy[:, 1]

print('solution has minimum %.6e, maximum %.6e' % (u.min(),u.max()))

if (args.contours == np.nan):
    NC = 10
    print('plotting with %d automatic contours ...' % NC)
    du = u.max() - u.min()
    C = np.linspace(u.min()+du/(2*NC),u.max()-du/(2*NC),NC)
else:
    C = np.array(args.contours)

plt.figure()
plt.gca().set_aspect('equal')
plt.tricontour(x, y, ele, u, C, colors='k', extend='neither', linewidths=0.5, linestyles='solid')
plt.axis('off')

if len(args.o) > 0:
    print('saving contour map in %s ...' % args.o)
    plt.savefig(args.o,bbox_inches='tight')
else:
    print('plotting to screen (-o not given) ...')
    plt.show()

