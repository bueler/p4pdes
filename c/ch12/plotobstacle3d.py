#!/usr/bin/env python

help = '''
Plot solution and obstacle in 3D figure.  Reads output from obstacle using
option
   -obs_dump_binary foo.dat
Requires access to
   petsc/petsc/lib/petsc/bin/PetscBinaryIO.py
   petsc/petsc/lib/petsc/bin/petsc_conf.py
e.g. sym-links.
'''

# generate figure in book by:
#   ./obstacle -da_refine 5 -obs_dump_binary obstacle65.dat
#   ./plotobstacle3d.py -o obstacle65.pdf -mx 65 -my 65 obstacle65.dat
#   pdfcrop obstacle65.pdf obstacle65.pdf

from sys import exit
from argparse import ArgumentParser, RawTextHelpFormatter

parser = ArgumentParser(description=help,
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('file',metavar='DATFILE',default='',
                    help='binary file saved from -obs_dump_binary DATFILE')
parser.add_argument('-mx',metavar='MX', type=int, default=-1,
                    help='spatial grid with MX points in x direction')
parser.add_argument('-my',metavar='MY', type=int, default=-1,
                    help='spatial grid with MY points in y direction')
parser.add_argument('-o',metavar='OUTFILE',default='',
                    help='image file such as .png')
args = parser.parse_args()

import PetscBinaryIO
import numpy as np
import matplotlib.pyplot as plt

io = PetscBinaryIO.PetscBinaryIO()
fh = open(args.file)
objecttype = io.readObjectType(fh)
if objecttype == 'Vec':
    uin = io.readVec(fh)
else:
    print 'error reading first Vec'
    sys.exit(1)
u = np.reshape(uin,(args.my,args.mx))
objecttype = io.readObjectType(fh)
if objecttype == 'Vec':
    psiin = io.readVec(fh)
else:
    print 'error reading second Vec'
    sys.exit(1)
psi = np.reshape(psiin,(args.my,args.mx))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

# plot entire unit sphere (in grey)
theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 100)
xs = np.outer(np.cos(theta), np.sin(phi))
ys = np.outer(np.sin(theta), np.sin(phi))
zs = np.outer(np.ones(np.size(theta)), np.cos(phi))
ax.plot_surface(xs, ys, zs, color='grey', alpha=1.0)

# plot z = u(x,y) over sphere
x = np.linspace(-2.0,2.0,args.mx)
y = np.linspace(-2.0,2.0,args.my)
xx, yy = np.meshgrid(x,y)
ax.plot_wireframe(xx,yy,u,color='k',linewidth=0.3)

plt.axis('off')

if len(args.o) > 0:
    print 'writing file %s' % args.o
    plt.savefig(args.o, dpi=fig.dpi)
else:
    plt.show()

