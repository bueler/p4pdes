#!/usr/bin/env python
#
# (C) 2018 Ed Bueler

# example:

from __future__ import print_function
import sys, argparse

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='''
Convert GAMG interpolation matrix into .tikz figure showing aggregates.
Input file has PETSc binary representation of the Mat.  The .tikz
can be included into LaTeX documents.  Requires ../PetscBinaryIO.py.
''')
# positional filename:
parser.add_argument('inname', metavar='NAME', default='',
                    help='input file name for file containing one Mat in PETSc binary format')
args = parser.parse_args()

import numpy as np

sys.path.append('../')
import PetscBinaryIO # may need link

io = PetscBinaryIO.PetscBinaryIO()
print('  reading Mat from %s ...' % args.inname)
fh = open(args.inname)
objecttype = io.readObjectType(fh)
if objecttype == 'Mat':
    Pint = io.readMatDense(fh)
    print(np.shape(Pint))
else:
    print('ERROR: no valid .Mat at start of file ... stopping')
    sys.exit(1)

fh.close()

