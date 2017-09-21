#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser(description=
"""
Generate residual norm figure from PETSc monitor output from a SNES+KSP solve.
Can show SNES or KSP norms only.  Can show true residual norms from KSP.

Example:
  $ ./program -snes_monitor -ksp_monitor &> foo.txt
  $ ./sneskspplot.py foo.txt
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('infile', metavar='NAME', help='text input file', default='foo')
parser.add_argument('--color', action='store_true',
                    help='use color in plot', default=False)
parser.add_argument('--ksptrue', action='store_true',
                    help='read output from -ksp_monitor_true_residual', default=False)
parser.add_argument('--markersize', dest='msize',
                    help='use this markersize for major symbols', default=10.0)
parser.add_argument('--noksp', action='store_false', 
                    help='do not show KSP residuals', dest='showksp',default=True)
parser.add_argument('--nosnes', action='store_false',
                    help='do not show SNES residuals', dest='showsnes',default=True)
parser.add_argument('-o', help='output filename (using matplotlib.pyplot.savefig; e.g. .png,.pdf)', dest='output')
parser.add_argument('--showdata', action='store_true',
                    help='print raw list of norms', default=False)

args = parser.parse_args()

infile = open(args.infile, 'r')

data = []
for line in infile:
    # strip comment lines starting with "#"
    line = line.partition('#')[0]
    line = line.rstrip()
    if line: # if content remains
        ls = line.split()
        for k in range(len(ls)):
            if ls[k] == 'SNES' and ls[k+1] == 'Function' and ls[k+2] == 'norm':
                norm = float(ls[k+3])
                data.append([norm])
            if args.showksp:
                if args.ksptrue:
                    if ls[k] == 'true' and ls[k+1] == 'resid' and ls[k+2] == 'norm':
                        norm = float(ls[k+3])
                        data[-1].append(norm)
                else:
                    if ls[k] == 'KSP' and ls[k+1] == 'Residual' and ls[k+2] == 'norm':
                        norm = float(ls[k+3])
                        data[-1].append(norm)

if args.showdata:
    print data

fig = plt.figure(figsize=(7,6))

# determine symbols
if args.color:
    mcolor = 'b'
else:
    mcolor = 'k'
snesmarker = mcolor + 'o'
kspmarker = mcolor + '*'
if not args.showsnes:
    kspmarker = snesmarker

# actually plot points
if args.ksptrue:
    ksplabel = 'KSP true residual norm'
else:
    ksplabel = 'KSP residual norm'
for k in range(len(data)):
    if args.showsnes:
        if k == 0:
            plt.semilogy(k,data[k][0],snesmarker,markersize=args.msize,label='SNES residual norm')
        else:
            plt.semilogy(k,data[k][0],snesmarker,markersize=args.msize)
    nksp = len(data[k]) - 1          # number of ksp markers to show
    if args.showksp and nksp > 0:
        jj = np.arange(nksp) + 1
        jjshift = k + jj / float(nksp+1)
        if k == 0:
            plt.semilogy(jjshift,data[k][1:],kspmarker+'-',markersize=args.msize-2,label=ksplabel)
        else:
            plt.semilogy(jjshift,data[k][1:],kspmarker+'-',markersize=args.msize-2)

plt.grid(True)
plt.xlabel('SNES iteration',fontsize=18.0)
plt.xticks(range(len(data)))

if args.showksp:
    plt.legend(loc='upper right')

if args.output:
    plt.savefig(args.output,bbox_inches='tight')
else:
    plt.show()

