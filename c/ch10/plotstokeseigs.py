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

def writeout(fname):
    print 'writing image file %s ...' % fname
    plt.savefig(fname,bbox_inches='tight')

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
face = ['k','w']
res = ['2','3','4']
preambles = ['neg','pos']
for negpos in range(2):
    plt.figure()
    for q in range(2):
       for r in range(3):
           MM = readcsr(fnames[q]+res[r]+'.dat').todense()
           lam, v = eig(MM)
           lam = np.sort(np.real(lam))
           if negpos == 0:
               lam = lam[lam < 0.0]
           else:
               lam = lam[lam >= 0.0]
           print '%s eigenvalues of %s at res %s:' % \
                 (preambles[negpos],fnames[q],res[r])
           print lam
           plt.plot(lam,np.zeros(np.shape(lam))+(1-q)+0.2*(2-r),
                    'o',color='k',markerfacecolor=face[q])
    plt.ylim(-0.6,1.8)
    if negpos == 1:
        plt.xlim(0.0,6.0)
    plt.grid(True)
    plt.gca().set_yticks([0.0,0.2,0.4,1.0,1.2,1.4])
    plt.gca().set_yticklabels(['m=33','m=17','m=9','m=33','m=17','m=9'])
    writeout(preambles[negpos]+'eigs.pdf')

