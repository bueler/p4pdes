#!/usr/bin/env python

# generate figures showing negative and positive eigenvalues for the staggered and regular methods

# run:
#   make well
#   ./mats1D.sh
#   ./plotstokeseigs.py  # FIXME:  file names as arguments

import PetscBinaryIO

import numpy as np
import matplotlib.pyplot as plt

import sys

from scipy.sparse import csr_matrix
from scipy.linalg import eig

# row indices in PETSc MatSparse need filling-in to be used by csr_matrix()
def fillrowindices(I,J,M):
    II = np.zeros(len(J))
    for k in range(M):
       II[I[k]:I[k+1]] = k
    return II

# read a PETSc Mat, stored in PETSc binary format, into a scipy.sparse format
def readcsr(filename):
    io = PetscBinaryIO.PetscBinaryIO()
    try:
        fh = open(filename)
    except IOError:
        print "Could not read file ", filename
        sys.exit()
    objecttype = io.readObjectType(fh)
    if objecttype == 'Mat':
        #from PetscBinaryIO.py source:
        #   M,N : matrix size
        #   I,J : arrays of row and column for each nonzero
        #   V: nonzero values (matrix entries)
        (M,N), (I,J,V) = io.readMatSparse(fh)
        II = fillrowindices(I,J,M)
        return csr_matrix((V, (II,J)), shape=(M,N))
    else:
        print "ERROR reading objecttype Mat from ", filename, "; got ", objecttype
        sys.exit()

fnames = ['matstag','matregu']
preambles = ['neg','pos']
for negpos in range(2):
    plt.figure()
    for q in range(2):
       MM = readcsr(fnames[q]+'.dat').todense()
       lam, v = eig(MM)
       lam = np.sort(np.real(lam))
       if negpos == 0:
           lam = lam[lam < 0.0]
       else:
           lam = lam[lam >= 0.0]
       print lam
       plt.plot(lam,np.zeros(np.shape(lam))+(1-q),'o',color='k')
    plt.ylim(-0.5,1.5)
    plt.grid(True)
    plt.gca().set_yticks([0.0,1.0])
    plt.gca().set_yticklabels(['regular','staggered'])
    plt.savefig(preambles[negpos]+'eigs.pdf',bbox_inches='tight')   # FIXME print filename at write

