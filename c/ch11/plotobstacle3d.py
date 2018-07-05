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

#FIXME  need 3D graphics
plt.imshow(u)
#plt.xlabel('t')
if len(args.o) > 0:
    print 'writing file %s' % args.o
    plt.savefig(args.o)
else:
    plt.show()

