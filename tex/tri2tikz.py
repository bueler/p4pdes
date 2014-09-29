#!/usr/bin/env python
#
# (C) 2014 Ed Bueler
#
# Create a tikz figure from the .node, .ele, .poly output of triangle.

import numpy
import argparse
import sys

commandline = " ".join(sys.argv[:])

parser = argparse.ArgumentParser(description='Converts .node, .ele, .poly files from triangle into TikZ format.')
parser.add_argument('--labelnodes', action='store_true',
                    help='label the nodes with zero-based index',
                    default=False)
parser.add_argument('--polyonly', action='store_true',
                    help='build a TikZ figure with the polygon only (reads .poly and .node)',
                    default=False)
parser.add_argument('--scale', action='store', metavar='X',
                    help='amount by which to scale TikZ figure',
                    default=1.0)
parser.add_argument('--labeloffset', action='store', metavar='X',
                    help='offset to use in labeling points',
                    default=0.0)
parser.add_argument('inroot', metavar='NAMEROOT',
                    help='root of input file name for .node,.ele,.poly',
                    default='foo')
parser.add_argument('outfile', metavar='FILENAME',
                    help='output file name',
                    default='foo.tikz')

args = parser.parse_args()
nodename = args.inroot + '.node'
elename  = args.inroot + '.ele'
polyname = args.inroot + '.poly'
outname  = args.outfile
dolabel  = args.labelnodes
polyonly = args.polyonly
scale    = (float)(args.scale)
offset   = (float)(args.labeloffset)

# always read .poly and .node files
polyfile = open(polyname, 'r')
nodefile = open(nodename, 'r')
if not polyonly:
  elefile = open(elename,  'r')

tikz = open(outname, 'w')

print 'reading from %s ' % polyname
polyheader1 = polyfile.readline()
polyheader2 = polyfile.readline()
print '  header says (line 1):  ' + polyheader1,
print '  header says (line 2):  ' + polyheader2,
P = (int)(polyheader2.split()[0])
print '  will read P=%d boundary segments ...' % P

print 'reading from %s ' % nodename
nodeheader = nodefile.readline()
print '  header says:  ' + nodeheader,
N = (int)(nodeheader.split()[0])
print '  will read N=%d nodes ...' % N

# read node info into arrays
x = numpy.zeros(N)
y = numpy.zeros(N)
bt = numpy.zeros(N,dtype=numpy.int32)
for j in range(N):
  nxyb = nodefile.readline()
  if nxyb == '':
    break
  if nxyb == '\n':
    continue
  nxyb = nxyb.split()
  if nxyb[0] == '#':
    continue
  if j+1 != int(nxyb[0]):
    print 'ERROR: INDEXING WRONG IN READING NODES'
  x[j] = float(nxyb[1])
  y[j] = float(nxyb[2])
  bt[j] = int(nxyb[3])

if not polyonly:
  print 'reading from %s ' % elename
  eleheader = elefile.readline()
  print '  header says:  ' + eleheader,
  K = (int)(eleheader.split()[0])
  print '  will read K=%d elements ...' % K

print 'writing %s ...' % outname,

tikz.write('%% created by script tri2tikz.py command line:%s\n' % '')
tikz.write('%%   %s\n' % commandline)
tikz.write('%%%s\n' % '')
tikz.write('\\begin{tikzpicture}[scale=%f]\n' % scale)

if not polyonly:
  # read element file (.ele) and plot elements with light gray
  for m in range(K):
    emabc = elefile.readline()
    if emabc == '':
      break
    if emabc == '\n':
      continue
    emabc = emabc.split()
    if emabc[0] == '#':
      continue
    if m+1 != int(emabc[0]):
      print 'ERROR: INDEXING WRONG IN READING ELEMENTS'
    pp = [int(emabc[1])-1, int(emabc[2])-1, int(emabc[3])-1] # node indices for this element
    kk = [0, 1, 2, 0]  # cycle through nodes
    for k in range(3):
      jfrom = pp[kk[k]]
      jto   = pp[kk[k+1]]
      tikz.write('  \\draw[gray,very thin] (%f,%f) -- (%f,%f);\n' \
                 % (x[jfrom],y[jfrom],x[jto],y[jto]))
  # plot nodes themselves
  for j in range(N):
    if dolabel:
      tikz.write( '  \\draw (%f,%f) node {$%d$};\n' % (x[j]+0.7*offset,y[j]-offset,j))
    tikz.write('  \\filldraw (%f,%f) circle (1.25pt);\n' % (x[j],y[j]))

# read polygon file (.poly) and plot according to boundary type
for p in range(P):
  pjkb = polyfile.readline()
  if pjkb == '':
    break
  if pjkb == '\n':
    continue
  pjkb = pjkb.split()
  if pjkb[0] == '#':
    continue
  if p+1 != int(pjkb[0]):
    print 'ERROR: INDEXING WRONG IN READING ELEMENTS'
  jfrom = int(pjkb[1])-1
  jto   = int(pjkb[2])-1
  bt    = int(pjkb[3])
  if bt == 2:
    mywidth = '2.5pt' # strong line for Dirichlet part
  else:
    mywidth = '0.75pt'
  tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
             % (mywidth,x[jfrom],y[jfrom],x[jto],y[jto]))

tikz.write('\\end{tikzpicture}\n')
tikz.close()
print 'done'

polyfile.close()
nodefile.close()
if not polyonly:
  elefile.close()

