# (C) 2018-2020 Ed Bueler

# See msh2petsc.py for usage.

# This script is based on the Gmsh ASCII format (version 4.1) documented at
#   http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format

import numpy as np
import sys


#Gmsh format version 4.1:
#$Entities
#  numPoints numCurves numSurfaces numVolumes      # use: numPoints, numCurves
#                                                  # check: numSurfaces = 1, numVolumes = 0
#  pointTag X Y Z numPhysicalTags                  # check: numPhysicalTags = 0
#  ...
#  curveTag minX minY minZ maxX maxY maxZ numPhysicalTags physicalTag
#    numBoundingPoints pointTag ...                # use: curveTag, physicalTag
#                                                  # check: numPhysicalTags = 1
#  ...
#  surfaceTag minX minY minZ maxX maxY maxZ numPhysicalTags physicalTag
#    numBoundingCurves curveTag ...                # use: surfaceTag, physicalTag
#                                                  # check: numPhysicalTags = 1
#  volumeTag ...                                   # should be absent
#$EndEntities

def read_entities_41(filename):
    Entitiesread = False
    EndEntitiesread = False
    firstlineread = False
    Npoints = 0
    Ncurves = 0
    count = 0
    tagmap = {}    # dictionary:  tagmap[curveTag] = physicalTag
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Entities':
                    assert (not Entitiesread), '$Entities repeated'
                    Entitiesread = True
                elif line == '$EndEntities':
                    assert (Entitiesread), '$EndEntities before $Entities'
                    break  # apparent success reading the entities
                elif Entitiesread:
                    ls = line.split(' ')
                    assert (len(ls) in [4,5,12,14]), 'unexpected line format'
                    if len(ls) == 4:
                        assert (not firstlineread), 'only one length 4 line expected'
                        Npoints = int(ls[0])
                        Ncurves = int(ls[1])
                        assert (int(ls[2]) == 1), 'expected only one surface entity'
                        assert (int(ls[3]) == 0), 'expected zero volume entities'
                        firstlineread = True
                    elif len(ls) == 5:
                        continue
                    elif len(ls) == 12:
                        assert (int(ls[7]) == 1), 'expected only one physical tag per curve'
                        tagmap[int(ls[0])] = int(ls[8])
                    elif len(ls) == 14:
                        assert (int(ls[7]) == 1), 'expected only one physical tag per surface'
                        tagmap[int(ls[0])] = int(ls[8])
    return tagmap

#Gmsh format version 4.1:
#$Nodes
#  numEntityBlocks numNodes minNodeTag maxNodeTag    # use: numNodes
#  entityDim entityTag parametric numNodesInBlock    # use: numNodesInBlock
#                                                    # check: parametric == 0
#    nodeTag
#    ...
#    x(double) y(double) z(double)                   # ignore z
#    ...
#  ...
#$EndNodes

def read_nodes_41(filename):
    Nodesread = False
    EndNodesread = False
    firstlineread = False
    N = 0           # number of nodes
    count = 0       # count of nodes read
    blocksize = 0   # number of nodes in block
    blocknodecount = 0    # count of node tags read in block
    nodetag = []    # node tag as read
    coords = []     # pairs (x-coord, y-coord)
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Nodes':
                    assert (not Nodesread), '$Nodes repeated'
                    Nodesread = True
                elif line == '$EndNodes':
                    assert (Nodesread), '$EndNodes before $Nodes'
                    assert (len(coords) >= 2), '$EndNodes reached before any nodes read'
                    break  # apparent success reading the nodes
                elif Nodesread:
                    ls = line.split(' ')
                    assert (len(ls) in [1,3,4]), 'unexpected line format'
                    if len(ls) == 4:
                        if not firstlineread:
                            N = int(ls[1])
                            firstlineread = True
                            coords = np.zeros(2*N)            # allocate space for coordinates
                        else:
                            assert (N > 0), 'N not defined'
                            assert (ls[2] == '0'), 'parametric not equal to zero'
                            blocksize = int(ls[3])
                            assert (blocksize <= N - count), 'expected to read fewer nodes'
                            blocknodecount = 0
                            blockcoordscount = 0
                        continue
                    elif len(ls) == 1:
                        assert (firstlineread), 'first line of nodes not yet read'
                        assert (blocknodecount < blocksize), 'not expecting a node tag'
                        thistag = int(ls[0])
                        nodetag.append(thistag)
                        blocknodecount += 1
                    elif len(ls) == 3:
                        assert (firstlineread), 'first line of nodes not yet read'
                        xy = [float(s) for s in ls[0:2]]
                        count += 1
                        coords[2*(count-1):2*count] = xy
    assert (count == N), 'N does not agree with count'
    return N,coords,nodetag


#Gmsh format version 4.1:
#$Elements
#  numEntityBlocks numElements minElementTag maxElementTag   # use: numElements
#  entityDim entityTag elementType numElementsInBlock        # use: entityTag, elementType, numElementsInBlock
#    elementTag nodeTag nodeTag                   # when elementType == 1
#    elementTag nodeTag nodeTag nodeTag           # when elementType == 2
#    ...
#  ...
#$EndElements

def read_elements_41(filename,N,nodetag,tagmap):
    Elementsread = False
    firstlineread = False
    NE = 0
    count = 0           # count of elements read
    blocksize = 0       # number of elements in block
    blockentity = 0     # entityTag which applies for block
    blocktype = 0       # =1 for boundary segments, =2 for triangles
    blockcount = 0      # count of elements read in block
    tri = []
    bf = np.zeros(N,dtype=int)   # zero for interior
    ns = []
    with open(filename, 'r') as mshfile:
        for line in mshfile:
            line = line.strip()  # remove leading and trailing whitespace
            if line: # only look at nonempty lines
                if line == '$Elements':
                    assert (not Elementsread), '"$Elements" repeated'
                    Elementsread = True
                elif line == '$EndElements':
                    assert (Elementsread), '"$EndElements" before "$Elements"'
                    assert (len(tri) > 0), 'no triangles read'
                    break  # apparent success
                elif Elementsread:
                    ls = line.split(' ')
                    assert (len(ls) in [3,4]), 'unexpected line format'
                    if len(ls) == 4:
                        if not firstlineread:
                            NE = int(ls[1])
                            firstlineread = True
                        elif blockcount == blocksize:  # a line with 4 which describes the block
                            assert (NE > 0), 'NE not defined'
                            blockentity = int(ls[1])
                            blocktype = int(ls[2])
                            blocksize = int(ls[3])
                            assert (blocksize <= NE - count), 'expected to read fewer elements'
                            blockcount = 0
                        else:  # read a triangle
                            assert (blocktype == 2), 'expecting a triangle'
                            assert (blockcount < blocksize), 'already read all elements in block'
                            thistri = [nodetag.index(int(s)) for s in ls[1:4]]
                            tri.append(np.array(thistri,dtype=int))
                            blockcount += 1
                            count += 1
                    else:      # read a boundary segment
                        assert (blocktype == 1), 'expecting a boundary segment'
                        assert (blockcount < blocksize), 'already read all elements in block'
                        thisbs = [nodetag.index(int(s)) for s in ls[1:3]]
                        for j in range(2):
                            # Dirichlet=2 wins for nodes
                            bf[thisbs[j]] = max(bf[thisbs[j]],tagmap[blockentity])
                        if tagmap[blockentity] == 1:
                            ns.append(np.array(thisbs,dtype=int))
                        blockcount += 1
                        count += 1
    assert (count == NE), 'count of elements read does not equal numElements'
    return np.array(tri).flatten(),bf,np.array(ns).flatten()

