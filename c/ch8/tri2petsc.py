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
#    * test output with a minimal PETSc code that does nothing more than
#      VecLoad(), VecView(), ISLoad(), ISView()
#    * decide on how boundary edges will be indicated

import numpy as np
import sys

# debug print
VERBOSITY=0;  # set to large value to see messages
def dprint(d,s):
    if d < VERBOSITY:
        print s

# N,x,y,bfn = triangle_read_node(filename)
#     N = number of nodes
#     x = x coordinates of nodes
#     y = y coordinates of nodes
#     bfn = integer flag for boundary nodes; 2 = dirichlet, 3 = neumann
def triangle_read_node(filename):
    nodefile = open(filename, 'r')
    headersread = 0
    count = 0
    N = 0
    x = []
    y = []
    bfn = []
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
                bfn = np.zeros(N,dtype=np.int32)
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
            bfn[count] = int(nxyb[3])
            count += 1
    nodefile.close()
    return N,x,y,bfn

# K,e,xc,yc = triangle_read_ele(filename,x,y)
#   in:
#     filename
#     x = x coordinate of node
#     y = y coordinate of node
#   out:
#     K = number of elements
#     e = K x 3 integer array of node indices
#     xc = x coordinate of element centroid
#     yc = y coordinate of element centroid
def triangle_read_ele(filename,x,y):
    elefile  = open(filename,  'r')
    headersread = 0
    count = 0
    K = 0
    e = []
    xc = []
    yc = []
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
                xc = np.zeros(K)
                yc = np.zeros(K)
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
            pp = [int(emabc[1])-1, int(emabc[2])-1, int(emabc[3])-1]
            # compute element centroid
            kk = [0, 1, 2, 0]  # cycle through nodes
            XX = 0.0
            YY = 0.0
            for k in range(3):
                jj = pp[kk[k]]
                XX += x[jj]
                YY += y[jj]
            XX /= 3.0
            YY /= 3.0
            dprint(1,'element %d centered at (%f,%f):' % (count,XX,YY))
            # append this element
            e[count,:] = pp
            xc[count] = XX
            yc[count] = YY
            count += 1
    elefile.close()
    return K,e,xc,yc

# PN,PS,px,py,s,bfs = triangle_read_poly(filename)
#     PN = number of node locations (in .poly file)
#     PS = number of segment pairs
#     px = x coordinate of node; length PN
#     py = y coordinate of node; length PN
#     s = segment pairs; PS x 2
#     bfs = integer flag for boundary segments; length PS; 2 = dirichlet, 3 = neumann
def triangle_read_poly(filename):
    polyfile = open(filename, 'r')
    headersread = 0
    ncount = 0
    scount = 0
    PN = 0
    PS = 0
    px = []
    py = []
    s = []
    bfs = []
    for line in polyfile:
        # strip comment
        line = line.partition('#')[0]
        line = line.rstrip()
        if line: # only act if line content remains
            # read headers and allocate when we can
            if headersread == 0:
                dprint(1,'  header 1 says:  ' + line)
                PN = (int)(line.split()[0])
                if PN > 0:
                    dprint(1,'  reading PN = %d nodes on boundary ...' % PN)
                    px = np.zeros(PN)
                    py = np.zeros(PN)
                headersread += 1
                ncount = 0
                continue
            if (headersread == 1) and (ncount >= PN):
                dprint(1,'  header 2 says:  ' + line)
                PS = (int)(line.split()[0])
                if PS > 0:
                    dprint(1,'  reading PS = %d boundary segments (ignoring markers) ...' % PS)
                    s = np.zeros((PS,2),dtype=np.int32)
                    bfs = np.zeros(PS,dtype=np.int32)
                headersread += 1
                scount = 0
                continue
            if (headersread == 2) and (scount >= PS):
                break # nothing more to read
            # read content if headersread = 1 or 2 and count is small
            if (headersread == 1) and (ncount < PN):
                # read a line describing a node on the boundary
                # note px[], py[] should exist
                nxyb = line.split()
                if ncount+1 != int(nxyb[0]):
                    print 'ERROR: INDEXING WRONG IN READING NODES ON POLYGON'
                    sys.exit(2)
                px[ncount] = float(nxyb[1])
                py[ncount] = float(nxyb[2])
                dprint(1,'  polygon node (%f,%f)' % (px[ncount],py[ncount]))
                ncount += 1
                continue
            if (headersread == 2) and (scount < PS):
                # read a line describing a boundary segment
                # note s[] should exist
                pjkb = line.split()
                if scount+1 != int(pjkb[0]):
                    print 'ERROR: INDEXING WRONG IN READING POLYGON SEGMENTS'
                    sys.exit(2)
                s[scount,:] = [int(pjkb[1])-1, int(pjkb[2])-1]
                bfs[scount] = int(pjkb[3])
                dprint(1,'polygon segment (%d,%d)' % tuple(s[scount,:]))
                scount += 1
            else:
                print 'ERROR:  headersread not 1 or 2 and yet trying to read; something wrong ... exiting'
                sys.exit(1)
    polyfile.close()
    return PN,PS,px,py,s,bfs


if __name__ == "__main__":
    import argparse
    import PetscBinaryIO
    # need link to petsc/bin/petsc-pythonscripts/PetscBinaryIO.py

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
    N,x,y,bfn = triangle_read_node(nodename)
    print '... N=%d nodes' % N

    elename = args.inroot + '.ele'
    print 'reading element triples from %s ' % elename,
    K,e,xc,yc = triangle_read_ele(elename,x,y)
    print '... K=%d elements' % K

    polyname = args.inroot + '.poly'
    print 'reading polygon from %s ' % polyname,
    PN,PS,px,py,s,bfs = triangle_read_poly(polyname)
    print '... PN=%d nodes and PS=%d segments' % (PN,PS)

    # FIXME presumably need to write more
    print 'writing node locations and element triples to petsc binary file %s' % args.outfile
    ox = x.view(PetscBinaryIO.Vec)
    oy = y.view(PetscBinaryIO.Vec)
    e = e.flatten()
    oe = e.view(PetscBinaryIO.IS)
    petsc = PetscBinaryIO.PetscBinaryIO()
    petsc.writeBinaryFile(args.outfile,[ox,oy,oe])

