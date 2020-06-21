# (C) 2018-2020 Ed Bueler

# See msh2petsc.py for usage.

# This module is based on the Gmsh ASCII format (version 2.2; legacy) at
#   http://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format-version-2-_0028Legacy_0029

import numpy as np
import sys

# fail and exit with message
def fail(s):
    print('ERROR: %s ... stopping' % s)
    sys.exit(1)

#Gmsh format version 2.2 (legacy):
#$Nodes
#number-of-nodes
#node-number x-coord y-coord z-coord       # ignore z-coord
#...
#$EndNodes

def read_nodes_22(filename):
    Nodesread = False
    EndNodesread = False
    N = 0   # number of nodes
    count = 0
    coords = []
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
                    if len(ls) == 1:
                        assert (N == 0), 'N found again but already read'
                        N = int(ls[0])
                        assert (N > 0), 'N invalid'
                        coords = np.zeros(2*N)  # allocate space for nodes
                    else:
                        assert (N > 0), 'expected to read N by now'
                        assert (len(ls) == 4), 'expected to read four values on node line'
                        rcount = int(ls[0])
                        count += 1
                        assert (count == rcount), 'unexpected (noncontiguous?) node indexing'
                        xy = [float(s) for s in ls[1:3]]
                        coords[2*(count-1):2*count] = xy            
    assert (count == N), 'N does not agree with index'
    return N,coords

#Gmsh format version 2.2 (legacy):
#$Elements
#number-of-elements
#elm-number elm-type number-of-tags < tag > ... node-number-list
#...
#$EndElements

def read_elements_22(filename,N,phys):
    Elementsread = False
    NE = 0   # number of Elements (in Gmsh sense; both triangles and boundary segments)
    tri = []
    ns = []
    bf = np.zeros(N,dtype=int)   # zero for interior
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
                    break  # apparent success reading the elements
                elif Elementsread:
                    ls = line.split(' ')
                    if len(ls) == 1:
                        assert (NE == 0), 'NE found again but already read'
                        NE = int(ls[0])
                        assert (NE > 0), 'NE invalid'
                    else:
                        assert (NE > 0), 'expected to read NE by now'
                        assert (len(ls) == 7 or len(ls) == 8), 'expected to read 7 or 8 values on element line'
                        dim = int(ls[1])
                        assert (dim == 1 or dim == 2), 'dim not 1 or 2'
                        etype = int(ls[3])
                        if dim == 2 and etype == phys['interior'] and len(ls) == 8:
                            # reading a triangle
                            thistri = [int(s) for s in ls[5:8]]
                            # change to zero-indexing
                            tri.append(np.array(thistri,dtype=int) - 1)
                        elif dim == 1 and len(ls) == 7:
                            ends = [int(s) for s in ls[5:7]]
                            if etype == phys['dirichlet']:
                                # reading a Dirichlet boundary segment; note zero-indexing
                                bf[np.array(ends,dtype=int) - 1] = 2
                            elif etype == phys['neumann']:
                                # reading a Neumann boundary segment; note zero-indexing
                                ns.append(np.array(ends,dtype=int) - 1)
                                ends = np.array(ends,dtype=int) - 1
                                for j in range(2):
                                    if bf[ends[j]] == 0:
                                        bf[ends[j]] = 1
                            else:
                                fail('should not be here: dim=1 and 7 entries but not etype')
                        else:
                            fail('should not be here: neither triangle or boundary segment')
    return np.array(tri).flatten(),bf,np.array(ns).flatten()

