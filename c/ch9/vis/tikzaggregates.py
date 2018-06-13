#!/usr/bin/env python
#
# (C) 2018 Ed Bueler

# example with 55 nodes and 6 aggregates:
#   ../unfem -un_mesh ../meshes/trap3 -pc_type gamg -un_gamg_save_pint_binary foo.mat
#   ./tikzaggregates.py -mat foo.mat -o foo.tikz ../meshes/trap3

from __future__ import print_function
import sys, argparse
from petsc2tikz import readcoordinates, readindices, writeelements
import numpy as np
sys.path.append('../')
import PetscBinaryIO # may need link

def writeselectnodes(tikz,N,xy,select,nodesize=1.25,labelnodes=False,nodeoffset=0.0):
    # plot nodes; looks better if *after* elements and segments
    for j in range(N):
        if abs(select[j]) > 0.0:
            tikz.write('  \\filldraw (%f,%f) circle (%fpt);\n' \
                       % (xy[j,0],xy[j,1],nodesize))
            if labelnodes:  # label node if desired
                tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                           % (xy[j,0]+0.7*nodeoffset,xy[j,1]-nodeoffset,j))

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='''
Convert GAMG interpolation matrix into .tikz figure showing aggregates.
Input -mat file has PETSc binary representation of the Mat.  Also needs
root for .vec,.is files in PETSc binary to describe the mesh.  The .tikz
can be included into LaTeX documents.  Requires ../PetscBinaryIO.py.
''')
parser.add_argument('-mat', metavar='MATNAME', default='',
                    help='input file name for file containing one Mat in PETSc binary format')
parser.add_argument('-o', metavar='FILENAME', default='',
                    help='output file name (.tikz or .tex)')
parser.add_argument('-scale', type=float, metavar='X', default=1.0,
                    help='amount by which to scale TikZ figure')
# positional filename:
parser.add_argument('meshroot', metavar='MESHROOT', default='',
                    help='root of file names for .vec,.is files containing mesh in PETSc binary format')
args = parser.parse_args()

io = PetscBinaryIO.PetscBinaryIO()

print('  reading Mat from %s ...' % args.mat)
fh = open(args.mat)
objecttype = io.readObjectType(fh)
if objecttype == 'Mat':
    Pint = io.readMatDense(fh)
    NPint, JJ = np.shape(Pint)
else:
    print('ERROR: no valid Mat at start of file ... stopping')
    sys.exit(1)
fh.close()
print('    found Mat of size (%d,%d)' % (NPint,JJ))

print('  reading node locations from %s ...' % (args.meshroot+'.vec'))
vecfile = open(args.meshroot+'.vec')
N, xy = readcoordinates(io,vecfile)
print('  reading topology from %s ...' % (args.meshroot+'.is'))
isfile  = open(args.meshroot+'.is')
K, P, e, bf, ns = readindices(io,isfile,N)
print('    found mesh with N=%d, K=%d, P=%d' % (N,K,P))

if NPint != N:
    print('ERROR: mismatched N values in .mat and .vec files ... stopping')
    sys.exit(2)

print('  writing to %s ...' % args.o)
tikz = open(args.o, 'w')
tikz.write('%% created by command line:%s\n' % '')
tikz.write('%%   %s\n' % commandline)
tikz.write('\\begin{tikzpicture}[scale=%f]\n' % args.scale)
writeelements(tikz,K,e,xy)
#FIXME
#for j in range(JJ):
#    writeselectnodes(tikz,N,xy,np.array(Pint[:,j]))
writeselectnodes(tikz,N,xy,np.array(Pint[:,0]))
tikz.write('\\end{tikzpicture}\n')
tikz.close()

