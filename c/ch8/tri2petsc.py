#!/usr/bin/env python
#
# (C) 2016 Ed Bueler
#
# Create a petsc binary file from the .node, .ele, .poly output of triangle.

# example to put Vec x,y (node coordinates) and IS e (element index map) in foo.dat:
#    $ cd tex/
#    $ make     # creates blob.1.* in c/ch8/meshes/
#    $ cd ../c/ch8/
#    $ ./tri2petsc.py meshes/blob.1 foo.dat

# TODO:
#    * test output with a minimal PETSc code that does nothing more than
#      VecLoad(), VecView(), ISLoad(), ISView()
#    * make print statements optional and verbosity dependent
#    * decide on how boundary edges will be indicated
#    * remove clutter

import numpy as np
import argparse
import sys

try:
    import PetscBinaryIO as pbio
except:
    print "'import PetscBinaryIO' failed"
    print "need link to petsc/bin/petsc-pythonscripts/PetscBinaryIO.py?"
    sys.exit(2)

try:
    import petsc_conf
except:
    print "'import petsc_conf.py' failed"
    print "need link to petsc/bin/petsc-pythonscripts/petsc_conf.py?"
    sys.exit(2)

if False:  # FIXME development clutter
    myvec = np.array([1., 2., 3.]).view(pbio.Vec)
    myis = np.array([9, 11, 15],dtype=np.int32).view(pbio.IS)
    io = pbio.PetscBinaryIO()
    io.writeBinaryFile('file.dat', [myvec,myis,])
    print 'file.dat written'
    fh = open('file.dat')
    while True:
        try:
            objecttype = io.readObjectType(fh)
        except:
            break
        if objecttype == 'Vec':
            v = io.readVec(fh)
            print 'vec: ',
            print v
        elif objecttype == 'IS':
            i = io.readIS(fh)
            print 'IS: ',
            print i
        else:
            print 'UNEXPECTED'
    print 'file.dat read'
    fh.close()
    sys.exit(0)

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='Converts .node, .ele, .poly files from triangle ASCII format into PETSc binary format.')
# positional filenames
parser.add_argument('inroot', metavar='NAMEROOT',
                    help='root of input file name for .node,.ele,.poly')
parser.add_argument('outfile', metavar='FILENAME',
                    help='output file name')

# process options: simpler names, and type conversions
args = parser.parse_args()
nodename = args.inroot + '.node'
elename  = args.inroot + '.ele'
polyname = args.inroot + '.poly'

# always read .poly file
polyfile = open(polyname, 'r')
nodefile = open(nodename, 'r')
elefile  = open(elename,  'r')

# debug print
VERBOSITY=99;
def dprint(d,s):
    if d < VERBOSITY:
        print s

# READ .node FILE
print 'reading from %s ' % nodename
headersread = 0
count = 0
for line in nodefile:
    # strip comment
    line = line.partition('#')[0]
    line = line.rstrip()
    if line: # only act if line content remains
        # read headers
        if headersread == 0:
            nodeheader = line
            print '  header says:  ' + nodeheader,
            N = (int)(nodeheader.split()[0])
            print '... reading N = %d nodes ...' % N
            x = np.zeros(N)
            y = np.zeros(N)
            bt = np.zeros(N,dtype=np.int32)
            headersread += 1
            continue
        elif (headersread == 1) and (count >= N):
            break # nothing more to read
        # read content if headersread = 1 and count is small
        nxyb = line.split()
        if count+1 != int(nxyb[0]):
            print 'ERROR: INDEXING WRONG IN READING NODES IN NODEFILE'
            sys.exit(2)
        x[count] = float(nxyb[1])
        y[count] = float(nxyb[2])
        bt[count] = int(nxyb[3])
        count += 1

print 'node coordinates:'
print x
print y

print 'node boundary markers:'
print bt

# READ .ele FILE
print 'reading from %s ' % elename
headersread = 0
count = 0
e = []
for line in elefile:
    # strip comment
    line = line.partition('#')[0]
    line = line.rstrip()
    if line: # only act if line content remains
        # read headers
        if headersread == 0:
            eleheader = line
            print '  header says:  ' + eleheader,
            K = (int)(eleheader.split()[0])
            print '... reading K = %d elements ...' % K
            headersread += 1
            continue
        elif (headersread == 1) and (count >= K):
            break # nothing more to read
        # read content if headersread = 1 and count is small
        emabc = line.split()
        if count+1 != int(emabc[0]):
            print 'ERROR: INDEXING WRONG IN READING ELEMENTS'
            sys.exit(2)
        # node indices for this element
        pp = [int(emabc[1])-1, int(emabc[2])-1, int(emabc[3])-1]
        # compute element centroid
        kk = [0, 1, 2, 0]  # cycle through nodes
        xc = 0.0
        yc = 0.0
        for k in range(3):
            jj = pp[kk[k]]
            xc = xc + x[jj]
            yc = yc + y[jj]
        xc = xc/3.0
        yc = yc/3.0
        # identify element and edges
        print 'element %d centered at (%f,%f):' % (count,xc,yc)
        for k in range(3):
            jfrom = pp[kk[k]]
            jto   = pp[kk[k+1]]
            print '  edge (%d,%d)' % (jfrom,jto),
            # FIXME: following test can mis-label an edge as BOUNDARY when it is
            #        not, in the case where its ends (nodes) are both in the boundary
            #        the solution seems to also require reading the .poly file so
            #        we know which segments are in the boundary
            if (bt[jfrom] > 0) and (bt[jto] > 0):
                if (bt[jfrom] == 2) and (bt[jto] == 2):
                    print ' DIRICHLET BOUNDARY'
                else:
                    print ' NEUMANN BOUNDARY'
            else:
                print ''
        # append this element
        e.append(pp)
        count += 1

e = np.array(e,dtype=np.int32).flatten()

polyfile.close()
nodefile.close()
elefile.close()

print 'done reading ...'
print ''

#FIXME need to write out indications that edges are in boundary
#FIXME possibly write out element centroid

print 'writing to petsc binary file %s ...' % args.outfile
ox = x.view(pbio.Vec)
oy = y.view(pbio.Vec)
oe = e.view(pbio.IS)
petsc = pbio.PetscBinaryIO()
petsc.writeBinaryFile(args.outfile, [ox,oy,oe])

