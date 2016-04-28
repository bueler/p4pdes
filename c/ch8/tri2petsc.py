#!/usr/bin/env python
#
# (C) 2016 Ed Bueler
#
# Create a petsc binary file from the .node, .ele, .poly output of triangle.

# example to put Vec x,y (node coordinates) and IS e (element index map) in foo.dat:
#    $ (cd tex/ && make)       # creates blob.1.* in c/ch8/meshes/
#    $ cd c/ch8/
#    $ ./tri2petsc.py meshes/blob.1 foo.dat

# TODO:
#    * create functions triangle_read_{poly,node,ele}() and set up as module
#      which can be used from tex/tri2tikz.py [avoids code duplication]
#    * test output with a minimal PETSc code that does nothing more than
#      VecLoad(), VecView(), ISLoad(), ISView()
#    * make print statements optional and verbosity dependent
#    * decide on how boundary edges will be indicated
#    * remove clutter

# FIXME development clutter
#    myvec = np.array([1., 2., 3.]).view(pbio.Vec)
#    myis = np.array([9, 11, 15],dtype=np.int32).view(pbio.IS)
#    io = pbio.PetscBinaryIO()
#    io.writeBinaryFile('file.dat', [myvec,myis,])
#    print 'file.dat written'
#    fh = open('file.dat')
#    while True:
#        try:
#            objecttype = io.readObjectType(fh)
#        except:
#            break
#        if objecttype == 'Vec':
#            v = io.readVec(fh)
#            print 'vec: ',
#            print v
#        elif objecttype == 'IS':
#            i = io.readIS(fh)
#            print 'IS: ',
#            print i
#        else:
#            print 'UNEXPECTED'
#    print 'file.dat read'
#    fh.close()
#    sys.exit(0)

import numpy as np
import sys

# debug print
VERBOSITY=0;  # set to large value to see messages
def dprint(d,s):
    if d < VERBOSITY:
        print s


def triangle_read_node(filename):
    nodefile = open(filename, 'r')
    headersread = 0
    count = 0
    for line in nodefile:
        # strip comment
        line = line.partition('#')[0]
        line = line.rstrip()
        if line: # only act if line content remains
            # read headers and allocate when you can
            if headersread == 0:
                nodeheader = line
                dprint(1,'  header says:  ' + nodeheader)
                N = (int)(nodeheader.split()[0])
                dprint(1,'  reading N = %d nodes ...' % N)
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
                print 'ERROR: INDEXING WRONG IN READING NODES'
                sys.exit(2)
            x[count] = float(nxyb[1])
            y[count] = float(nxyb[2])
            bt[count] = int(nxyb[3])
            count += 1
    nodefile.close()
    return N,x,y,bt


def triangle_read_ele(filename):
    elefile  = open(filename,  'r')
    headersread = 0
    count = 0
    e = []
    for line in elefile:
        # strip comment
        line = line.partition('#')[0]
        line = line.rstrip()
        if line: # only act if line content remains
            # read headers and allocate when you can
            if headersread == 0:
                eleheader = line
                dprint(1,'  header says:  ' + eleheader)
                K = (int)(eleheader.split()[0])
                dprint(1,'  reading K = %d elements ...' % K)
                xcentroid = np.zeros(K)
                ycentroid = np.zeros(K)
                e = np.zeros((K,3),dtype=np.int32)
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
            pp = np.array([int(emabc[1])-1, int(emabc[2])-1, int(emabc[3])-1],dtype=np.int32)
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
            dprint(1,'element %d centered at (%f,%f):' % (count,xc,yc))
            for k in range(3):
                jfrom = pp[kk[k]]
                jto   = pp[kk[k+1]]
                # FIXME: following test can mis-label an edge as BOUNDARY when it is
                #        not, in the case where its ends (nodes) are both in the boundary
                #        the solution seems to also require reading the .poly file so
                #        we know which segments are in the boundary
                bdrystr = ''
                if (bt[jfrom] > 0) and (bt[jto] > 0):
                    if (bt[jfrom] == 2) and (bt[jto] == 2):
                        bdrystr = ' DIRICHLET BOUNDARY'
                    else:
                        bdrystr = ' NEUMANN BOUNDARY'
                dprint(1,'  edge (%d,%d)%s' % (jfrom,jto,bdrystr))
            # append this element
            e[count,:] = pp
            xcentroid[count] = xc
            ycentroid[count] = yc
            count += 1
    elefile.close()
    return K,e,xc,yc


def triangle_read_poly(filename):
    polyfile = open(filename, 'r')
    headersread = 0
    count = 0
    bx = []
    by = []
    PN = 0
    PS = 0
    for line in polyfile:
        # strip comment
        line = line.partition('#')[0]
        line = line.rstrip()
        if line: # only act if line content remains
            # read headers and allocate when we can
            if headersread == 0:
                polyheader1 = line
                dprint(1,'  header 1 says:  ' + polyheader1)
                PN = (int)(polyheader1.split()[0])
                if PN > 0:
                    dprint(1,'  reading PN = %d nodes on boundary ...' % PN)
                    bx = np.zeros(PN)
                    by = np.zeros(PN)
                headersread += 1
                ncount = 0
                continue
            if (headersread == 1) and (ncount >= PN):
                polyheader2 = line
                dprint(1,'  header 2 says:  ' + polyheader2)
                PS = (int)(polyheader2.split()[0])
                if PS > 0:
                    dprint(1,'  reading PS = %d boundary segments (ignoring markers) ...' % PS)
                headersread += 1
                scount = 0
                continue
            if (headersread == 2) and (scount >= PS):
                break # nothing more to read
            # read content if headersread = 1 or 2 and count is small
            if (headersread == 1) and (ncount < PN):
                # read a line describing a node on the boundary; should only occur
                #   in cases where bx[], by[] exist
                nxyb = line.split()
                if ncount+1 != int(nxyb[0]):
                    print 'ERROR: INDEXING WRONG IN READING NODES ON POLYGON'
                    sys.exit(2)
                bx[ncount] = float(nxyb[1])
                by[ncount] = float(nxyb[2])
                dprint(1,'  polygon node (%f,%f)' % (bx[ncount],by[ncount]))
                scount += 1
                continue
            if (headersread == 2) and (scount < PS):
                # read a line describing a boundary segment
                pjkb = line.split()
                if scount+1 != int(pjkb[0]):
                    print 'ERROR: INDEXING WRONG IN READING POLYGON SEGMENTS'
                    sys.exit(2)
                jfrom = int(pjkb[1])-1
                jto   = int(pjkb[2])-1
                dprint(1,'polygon segment (%d,%d)' % (jfrom,jto))
                scount += 1
            else:
                print 'ERROR:  headersread not 1 or 2 and yet trying to read; something wrong ... exiting'
                sys.exit(1)
    polyfile.close()
    return PN,PS,bx,by


if __name__ == "__main__":
    import argparse
    import PetscBinaryIO
    # need link to petsc/bin/petsc-pythonscripts/PetscBinaryIO.py

    commandline = " ".join(sys.argv[:])

    parser = argparse.ArgumentParser(description= \
        'Converts .node, .ele, .poly files from triangle ASCII format into PETSc binary format.')
    # positional filenames
    parser.add_argument('inroot', metavar='NAMEROOT',
                        help='root of input file name for .node,.ele,.poly')
    parser.add_argument('outfile', metavar='FILENAME',
                        help='output file name')
    args = parser.parse_args()

    nodename = args.inroot + '.node'
    print 'reading nodes from %s ' % nodename,
    N,x,y,bt = triangle_read_node(nodename)
    print '... N=%d nodes' % N

    elename = args.inroot + '.ele'
    print 'reading element triples from %s ' % elename,
    K,e,xc,yc = triangle_read_ele(elename)
    print '... K=%d elements' % K

    polyname = args.inroot + '.poly'
    print 'reading polygon from %s ' % polyname,
    PN,PS,bx,by = triangle_read_poly(polyname)
    print '... PN=%d nodes and PS=%d segments' % (PN,PS)

    #FIXME need to write out indications that edges are in boundary
    #FIXME possibly write out element centroid

    print 'writing to petsc binary file %s' % args.outfile
    ox = x.view(PetscBinaryIO.Vec)
    oy = y.view(PetscBinaryIO.Vec)
    e = e.flatten()
    oe = e.view(PetscBinaryIO.IS)
    petsc = PetscBinaryIO.PetscBinaryIO()
    petsc.writeBinaryFile(args.outfile, [ox,oy,oe])

