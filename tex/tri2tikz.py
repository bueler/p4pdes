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
outname  = args.outfile
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
    N,x,y,bfn = triangle_read_node(nodename)
    print '... N=%d nodes' % N

    elename = args.inroot + '.ele'
    print 'reading element triples from %s ' % elename,
    K,e,xc,yc = triangle_read_ele(elename)
    print '... K=%d elements' % K

tikz = open(outname, 'w')
print 'writing to %s ...' % outname
tikz.write('%% created by script tri2tikz.py command line:%s\n' % '')
tikz.write('%%   %s\n' % commandline)
tikz.write('%%%s\n' % '')
tikz.write('\\begin{tikzpicture}[scale=%f]\n' % scale)

FIXME FROM HERE

# READ .node FILE BUT DO NO PLOTTING
if not polyonly:
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
                print '... reading N = %d nodes from interior and bdry ...' % N
                x = numpy.zeros(N)
                y = numpy.zeros(N)
                bt = numpy.zeros(N,dtype=numpy.int32)
                headersread += 1
                continue
            elif (headersread == 1) and (count >= N):
                break # nothing more to read
            # read content if headersread = 1 and count is small
            #DEBUG print 'reading nodefile content line with count = %d ...' % count
            nxyb = line.split()
            #DEBUG print nxyb
            if count+1 != int(nxyb[0]):
                print 'ERROR: INDEXING WRONG IN READING NODES IN NODEFILE'
                sys.exit(2)
            x[count] = float(nxyb[1])
            y[count] = float(nxyb[2])
            bt[count] = int(nxyb[3])
            count += 1

# READ .ele FILE AND PLOT TRIANGLES
if not polyonly:
    print 'reading from %s ' % elename
    headersread = 0
    count = 0
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
            #DEBUG print 'reading elefile content line with count = %d ...' % count
            emabc = line.split()
            #DEBUG print emabc
            if count+1 != int(emabc[0]):
              print 'ERROR: INDEXING WRONG IN READING ELEMENTS'
              sys.exit(2)
            pp = [int(emabc[1])-1, int(emabc[2])-1, int(emabc[3])-1] # node indices for this element
            kk = [0, 1, 2, 0]  # cycle through nodes
            for k in range(3):
              jfrom = pp[kk[k]]
              jto   = pp[kk[k+1]]
              if (not noboundary) or (bt[jfrom] == 0) or (bt[jto] == 0):
                tikz.write('  \\draw[gray,very thin] (%f,%f) -- (%f,%f);\n' \
                           % (x[jfrom],y[jfrom],x[jto],y[jto]))
            if dolabeleles:
              xc = 0.0
              yc = 0.0
              for k in range(3):
                jj = pp[kk[k]]
                xc = xc + x[jj]
                yc = yc + y[jj]
              xc = xc/3.0
              yc = yc/3.0
              tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                         % (xc+0.7*eleoffset,yc-eleoffset,count))
            count += 1

# READ .poly FILE
if not noboundary:
  print 'reading from %s ' % polyname
  headersread = 0
  count = 0
  for line in polyfile:
    # strip comment
    line = line.partition('#')[0]
    line = line.rstrip()
    if line: # only act if line content remains
      # read headers
      if headersread == 0:
          polyheader1 = line
          print '  header 1 says:  ' + polyheader1,
          NP = (int)(polyheader1.split()[0])
          if NP > 0:
            print '... reading NP = %d nodes on boundary ...' % NP
            # read node info into arrays
            bx = numpy.zeros(NP)
            by = numpy.zeros(NP)
          else:
            print '...'
          headersread += 1
          count = 0
          continue
      elif (headersread == 1) and (count >= NP):
          polyheader2 = line
          print '  header 2 says:  ' + polyheader2,
          P = (int)(polyheader2.split()[0])
          print '... reading P = %d boundary segments ...' % P
          headersread += 1
          count = 0
          continue
      elif (headersread == 2) and (count >= P):
          break # nothing more to read
      # read content if headersread = 1 or 2 and count is small
      #DEBUG  print 'reading content line with count = %d ...' % count
      if headersread == 1:
          # read a line describing a node on the boundary; should only occur
          #   in cases where bx[], by[] exist
          nxyb = line.split()
          if count+1 != int(nxyb[0]):
              print 'ERROR: INDEXING WRONG IN READING NODES ON POLYGON'
              sys.exit(2)
          bx[count] = float(nxyb[1])
          by[count] = float(nxyb[2])
          tikz.write('  \\filldraw (%f,%f) circle (%fpt);\n' % (bx[count],by[count],nodesize))
          count += 1
      elif headersread == 2:
          # read a line describing a boundary segment
          pjkb = line.split()
          if count+1 != int(pjkb[0]):
              print 'ERROR: INDEXING WRONG IN READING POLYGON SEGMENTS'
              sys.exit(2)
          jfrom = int(pjkb[1])-1
          jto   = int(pjkb[2])-1
          if int(pjkb[3]) == 2: # check boundary type
              mywidth = '2.5pt'   # strong line for Dirichlet part
          else:
              mywidth = '0.75pt'  # weak line for Neumann
          if polyonly:
              tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
                         % (mywidth,bx[jfrom],by[jfrom],bx[jto],by[jto]))
          else:
              tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
                         % (mywidth,x[jfrom],y[jfrom],x[jto],y[jto]))
          count += 1
      else:
          print 'ERROR:  headersread not 1 or 2 and yet trying to read; something wrong ... exiting'
          sys.exit(1)

if not polyonly:
  # plot interior and boundary nodes themselves
  for j in range(N):
    if dolabelnodes:
      tikz.write( '  \\draw (%f,%f) node {$%d$};\n' \
                 % (x[j]+0.7*nodeoffset,y[j]-nodeoffset,j))
    tikz.write('  \\filldraw (%f,%f) circle (%fpt);\n' % (x[j],y[j],nodesize))

tikz.write('\\end{tikzpicture}\n')
tikz.close()
print 'done'

polyfile.close()
if not polyonly:
  nodefile.close()
  elefile.close()

