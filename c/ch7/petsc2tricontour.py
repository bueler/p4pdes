#!/usr/bin/env python
#
# (C) 2016 Ed Bueler

#TODO:
#  * make output filename option
#  * put example in .sh script
#  * write up and put in book

# example:
#  ./kochmesh.py -l 5
#  triangle -pqa0.00002 koch.poly
#  showme koch
#  ./tri2petsc.py koch.1 koch.1
#  ./unfem -un_mesh koch.1 -un_case 4 -un_view_solution
#  ./petsc2tricontour.py -i koch.1 --contours 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.02 0.03 0.05 0.1

import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import PetscBinaryIO

parser = argparse.ArgumentParser(description='Read and show a contour plot of a solution on a triangulation.  Reads PETSc binary format files .vec,.is,.soln.')
parser.add_argument('-i', metavar='ROOT',
                    help='root of input file name for files .vec,.is,.soln')
parser.add_argument('--contours', metavar='C', type=np.double, nargs='+',
                    default=np.nan, help='contour levels')
args = parser.parse_args()

root = args.i

io = PetscBinaryIO.PetscBinaryIO()
vecfile  = open(root+'.vec')
isfile   = open(root+'.is')
solnfile = open(root+'.soln')

print 'reading triangulation from files %s,%s ...' % (root+'.vec',root+'.is')
objecttype = io.readObjectType(vecfile)
if objecttype == 'Vec':
    xy = io.readVec(vecfile)
    N = len(xy)
    if N % 2 != 0:
        print 'nodes in .vec file invalid'
        sys.exit()
    N /= 2
    xy = np.reshape(xy,(N,2))
else:
    print 'no valid .vec file'
    sys.exit()
objecttype = io.readObjectType(isfile)
if objecttype == 'IS':
    ele = io.readIS(isfile)
    K = len(ele)
    if K % 3 != 0:
        print 'elements (triangles) in .is file invalid'
        sys.exit()
    K /= 3
    if (ele.max() > N) or (ele.min() < 0):
        print 'elements contain invalid indices'
        sys.exit()
    ele = np.reshape(ele,(K,3))
else:
    print 'no valid .is file'
    sys.exit()

print 'reading solution from file %s ...' % (root+'.soln')
objecttype = io.readObjectType(solnfile)
if objecttype == 'Vec':
    u = io.readVec(solnfile)
    if len(u) != N:
        print 'solution vec is wrong size'
        sys.exit()
else:
    print 'no valid .soln file'
    sys.exit()

x = xy[:, 0]
y = xy[:, 1]

print 'solution has minimum %.6e, maximum %.6e ... plotting ...' % (u.min(),u.max())

if (args.contours == np.nan):
    NC = 10
    print 'plotting with %d automatic contours ...' % NC
    du = u.max() - u.min()
    C = np.linspace(u.min()+du/(2*NC),u.max()-du/(2*NC),NC)
else:
    C = np.array(args.contours)

plt.figure()
plt.gca().set_aspect('equal')
plt.tricontour(x, y, ele, u, C, colors='k', extend='neither', linewidth=2.0, linestyles='solid')
plt.axis('off')

plt.show()
#plt.savefig('foo.pdf',bbox_inches='tight')

