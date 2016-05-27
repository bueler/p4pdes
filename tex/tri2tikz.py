#!/usr/bin/env python
#
# (C) 2016 Ed Bueler
#
# Create a tikz figure from the .node, .ele, .poly output of triangle,
# or just from a .poly file (i.e. an input into triangle).

import numpy as np
import argparse
import sys

from tri2petsc import triangle_read_node, triangle_read_ele, triangle_read_poly
# need link to p4pdes/c/ch8/tri2petsc.py

commandline = " ".join(sys.argv[:])
parser = argparse.ArgumentParser(description='Converts .node, .ele, .poly files from triangle into TikZ format.')
# boolean options
parser.add_argument('--labelnodes', action='store_true',
                    help='label the nodes with zero-based index',
                    default=False)
parser.add_argument('--labelelements', action='store_true',
                    help='label the nodes with zero-based index',
                    default=False)
parser.add_argument('--polyonly', action='store_true',
                    help='use only the original polygon (.poly) file',
                    default=False)
parser.add_argument('--noboundary', action='store_true',
                    help='do not show the boundary segments at all',
                    default=False)
# real options
parser.add_argument('--scale', action='store', metavar='X',
                    help='amount by which to scale TikZ figure',
                    default=1.0)
parser.add_argument('--nodesize', action='store', metavar='X',
                    help='size (in points) for dots showing nodes',
                    default=1.25)
parser.add_argument('--nodeoffset', action='store', metavar='X',
                    help='offset to use in labeling nodes (points)',
                    default=0.0)
parser.add_argument('--eleoffset', action='store', metavar='X',
                    help='offset to use in labeling elements (triangles)',
                    default=0.0)
# positional filenames
parser.add_argument('inroot', metavar='NAMEROOT',
                    help='root of input file name for .node,.ele,.poly',
                    default='foo')
parser.add_argument('outfile', metavar='FILENAME',
                    help='output file name',
                    default='foo.tikz')

# process options: simpler names, and type conversions
args = parser.parse_args()
dolabelnodes = args.labelnodes
dolabeleles  = args.labelelements
polyonly     = args.polyonly
noboundary   = args.noboundary
scale        = (float)(args.scale)
nodesize     = (float)(args.nodesize)
nodeoffset   = (float)(args.nodeoffset)
eleoffset    = (float)(args.eleoffset)

polyname = args.inroot + '.poly'
print 'reading polygon from %s ' % polyname,
PN,PS,px,py,s,bfs = triangle_read_poly(polyname)
print '... PN=%d nodes and PS=%d segments' % (PN,PS)

if not polyonly:
    nodename = args.inroot + '.node'
    print 'reading nodes from %s ' % nodename,
    N,loc,bfn,L = triangle_read_node(nodename)
    print '... N=%d nodes with L=%d on Dirichlet bdry' % (N,L)

    elename = args.inroot + '.ele'
    print 'reading element triples from %s ' % elename,
    K,e,xc,yc = triangle_read_ele(elename,loc)
    print '... K=%d elements' % K

tikz = open(args.outfile, 'w')
print 'writing to %s ...' % args.outfile
tikz.write('%% created by script tri2tikz.py command line:%s\n' % '')
tikz.write('%%   %s\n' % commandline)
tikz.write('%%%s\n' % '')
tikz.write('\\begin{tikzpicture}[scale=%f]\n' % scale)

if not polyonly:
    # go through elements and draw edges and label centroids
    for ke in range(K):
        l = [0, 1, 2, 0]  # cycle through local node index
        for k in range(3):
            jfrom = e[ke,l[k]]
            jto   = e[ke,l[k+1]]
            if (not noboundary) or (bfn[jfrom] == 0) or (bfn[jto] == 0):
                tikz.write('  \\draw[gray,very thin] (%f,%f) -- (%f,%f);\n' \
                           % (loc[2*jfrom+0],loc[2*jfrom+1],loc[2*jto+0],loc[2*jto+1]))
            if dolabeleles:
                tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                           % (xc[ke]+0.7*eleoffset,yc[ke]-eleoffset,ke))
    # plot all nodes, with labels if wanted; looks better if *after* edges
    for j in range(N):
        tikz.write('  \\filldraw (%f,%f) circle (%fpt);\n' % (loc[2*j+0],loc[2*j+1],nodesize))
        if dolabelnodes:
            tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                       % (loc[2*j+0]+0.7*nodeoffset,loc[2*j+1]-nodeoffset,j))

if not noboundary:
    # go through boundary segments and plot with weight from type
    for js in range(PS):
        jfrom = s[js,0]
        jto = s[js,1]
        if bfs[js] == 2: # check boundary type
            mywidth = '2.5pt'   # strong line for Dirichlet part
        else:
            mywidth = '0.75pt'  # weak line for Neumann
        if polyonly:
            tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
                       % (mywidth,px[jfrom],py[jfrom],px[jto],py[jto]))
        else:
            tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
                       % (mywidth,loc[2*jfrom+0],loc[2*jfrom+1],loc[2*jto+0],loc[2*jto+1]))

tikz.write('\\end{tikzpicture}\n')
tikz.close()
print 'done'

