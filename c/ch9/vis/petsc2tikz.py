#!/usr/bin/env python
#
# (C) 2018 Ed Bueler

from __future__ import print_function
import sys, argparse

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='Convert .vec,.is unfem input files into .tikz for inclusion in LaTeX documents.  Requires ../PetscBinaryIO.py.')
parser.add_argument('--neumannonly', action='store_true', default=False,
                    help='only generate the Neumann boundary segments (no interior edges or nodes)')
parser.add_argument('--dirichletsize', type=float, metavar='X', default=2.5,
                    help='size (pt) for dots showing Dirichlet boundary nodes')
parser.add_argument('--eleoffset', type=float, metavar='X', default=0.0,
                    help='offset to use in labeling elements (triangles)')
parser.add_argument('--labelelements', action='store_true', default=False,
                    help='label the elements with zero-based index')
parser.add_argument('--labelnodes', action='store_true', default=False,
                    help='label the nodes with zero-based index')
parser.add_argument('--neumannwidth', type=float, metavar='X', default=2.5,
                    help='linewidth used in showing Neumann part of polygon')
parser.add_argument('--nodeoffset', type=float, metavar='X', default=0.0,
                    help='offset to use in labeling nodes (points)')
parser.add_argument('--nodesize', type=float, metavar='X', default=1.25,
                    help='size (pt) for dots showing nodes')
parser.add_argument('-o', metavar='FILENAME', default='',
                    help='output file name')
parser.add_argument('--scale', type=float, metavar='X', default=1.0,
                    help='amount by which to scale TikZ figure')
# positional filename:
parser.add_argument('inroot', metavar='NAMEROOT', default='',
                    help='root of input file name for .vec,.is')
args = parser.parse_args()
root = args.inroot

import numpy as np

sys.path.append('../')
import PetscBinaryIO # may need link

io = PetscBinaryIO.PetscBinaryIO()
vecfile = open(root+'.vec')
isfile  = open(root+'.is')

print('  reading node locations from %s ...' % (root+'.vec'))
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

print('  reading topology (element indices, boundary flags, Neumann segments) from %s ...' % (root+'.is'))
objecttype = io.readObjectType(isfile)
if objecttype == 'IS':
    e = io.readIS(isfile)
    K = len(e)
    if K % 3 != 0:
        print('ERROR: elements (triangles) in .is file invalid ... stopping')
        sys.exit()
    K /= 3
    if (e.max() >= N) or (e.min() < 0):
        print('ERROR: elements contain invalid indices ... stopping')
        sys.exit()
    e = np.reshape(e,(K,3))
    objecttype = io.readObjectType(isfile)
    bf = io.readIS(isfile)
    Nbf = len(bf)
    if Nbf != N:
        print('ERROR: number of boundary flags is wrong in .is file invalid ... stopping')
        sys.exit()
    if (bf.max() > 2) or (bf.min() < 0):
        print('ERROR: boundary flags contains invalid values ... stopping')
        sys.exit()
    objecttype = io.readObjectType(isfile)
    ns = io.readIS(isfile)
    if ns[0] >= 0:
        P = len(ns)
        if P % 2 != 0:
            print('ERROR: neumann segments in .is file invalid ... stopping')
            sys.exit()
        P /= 2
        if (ns.max() >= N) or (ns.min() < 0):
            print('ERROR: elements contain invalid indices ... stopping')
            sys.exit()
        ns = np.reshape(ns,(P,2))
    else:
        P = 0
        ns = []
else:
    print('ERROR: no valid .is file ... stopping')
    sys.exit()

print('  found: N=%d, K=%d, P=%d' % (N,K,P))

print('  writing to %s ...' % args.o)
tikz = open(args.o, 'w')
tikz.write('%% created by script tri2tikz.py command line:%s\n' % '')
tikz.write('%%   %s\n' % commandline)
tikz.write('\\begin{tikzpicture}[scale=%f]\n' % args.scale)

if not args.neumannonly:
    # go through elements: draw each triangle
    for ke in range(K):
        j = e[ke,0]
        tikz.write('  \\draw[gray,very thin] (%f,%f) ' % (xy[j,0],xy[j,1]))
        for k in [1, 2, 0]:  # cycle through local node index
            j = e[ke,k]
            tikz.write('-- (%f,%f) ' % (xy[j,0],xy[j,1]))
        tikz.write(';\n')
        if args.labelelements:  # label centroid if desired
            xc = np.average(xy[e[ke,:],0])
            yc = np.average(xy[e[ke,:],1])
            tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                       % (xc+0.7*args.eleoffset,yc-args.eleoffset,ke))
    # plot nodes; looks better if *after* edges
    for j in range(N):
        if bf[j] == 2:
            mysize = args.dirichletsize
        else:
            mysize = args.nodesize
        if mysize > 0:
            tikz.write('  \\filldraw (%f,%f) circle (%fpt);\n' % (xy[j,0],xy[j,1],mysize))
        if args.labelnodes:  # label node if desired
            tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                               % (xy[j,0]+0.7*args.nodeoffset,xy[j,1]-args.nodeoffset,j))

# go through Neumann boundary segments and plot with width
for js in range(P):
    jfrom = ns[js,0]
    jto = ns[js,1]
    mywidth = '%fpt' % args.neumannwidth
    tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
               % (mywidth,xy[jfrom,0],xy[jfrom,1],xy[jto,0],xy[jto,1]))

tikz.write('\\end{tikzpicture}\n')
tikz.close()

