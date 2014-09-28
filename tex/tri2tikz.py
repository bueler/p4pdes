#!/usr/bin/env python
#
# (C) 2014 Ed Bueler
#
# Create a tikz figure from the .node and .ele output of triangle.

import numpy
import argparse

parser = argparse.ArgumentParser(description='Converts .node and .ele files from triangle into TikZ format.')
parser.add_argument('--labelnodes', action='store_true',
                    help='label the nodes with zero-based index', default=False)
parser.add_argument('-n', '--nodefile', metavar='FILENAME', required = True,
                    help='input file name for nodes', default='foo.node')
parser.add_argument('-e', '--elefile', metavar='FILENAME', required = True,
                    help='input file name for elements', default='foo.ele')
parser.add_argument('-o', '--outfile', metavar='FILENAME', required = True,
                    help='output file name', default='foo.tikz')
args = parser.parse_args()

# FIXME use options on these:
scale = 1.0
offset = 0.25

nodefile = open(args.nodefile, 'r')
elefile = open(args.elefile, 'r')
tikz = open(args.outfile, 'w')

print 'reading from %s ' % args.nodefile,
nodeheader = nodefile.readline()
print '... header says:  ' + nodeheader,
N = (int)(nodeheader.split()[0])
print '  N=%d nodes' % N

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

print 'reading from %s ' % args.elefile,
eleheader = elefile.readline()
print '... header says:  ' + eleheader,
K = (int)(eleheader.split()[0])
print '  K=%d elements' % K

print 'writing %s ...' % args.outfile,

tikz.write('\\begin{tikzpicture}[scale=%f]\n' % scale)

for m in range(K):
  exyb = elefile.readline()
  if exyb == '':
    break
  if exyb == '\n':
    continue
  exyb = exyb.split()
  if exyb[0] == '#':
    continue
  if m+1 != int(exyb[0]):
    print 'ERROR: INDEXING WRONG IN READING ELEMENTS'
  p = [int(exyb[1])-1, int(exyb[2])-1, int(exyb[3])-1] # node indices for this element
  kk = [0, 1, 2, 0]  # cycle through nodes
  for k in range(3):
    jfrom = p[kk[k]]
    jto   = p[kk[k+1]]
    if bt[jfrom] > 0 and bt[jto] > 0:
      if bt[jfrom] == 2 and bt[jto] == 2:
        mywidth = '2.0pt'
      else:
        mywidth = '0.75pt'
      tikz.write('  \\draw[line width=%s] (%f,%f) -- (%f,%f);\n' \
                 % (mywidth,x[jfrom],y[jfrom],x[jto],y[jto]))
    else:
      tikz.write('  \\draw[gray,very thin] (%f,%f) -- (%f,%f);\n' \
                 % (x[jfrom],y[jfrom],x[jto],y[jto]))

for j in range(N):
  if args.labelnodes:
    tikz.write( '  \\draw (%f,%f) node {$%d$};\n' % (x[j]+0.7*offset,y[j]-offset,j))
  tikz.write('  \\filldraw (%f,%f) circle (1.25pt);\n' % (x[j],y[j]))

tikz.write('\\end{tikzpicture}\n')
print 'done'

nodefile.close()
elefile.close()
tikz.close()

