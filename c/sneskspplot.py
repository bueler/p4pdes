#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser(description=
"""
Generate residual norm figure from PETSc monitor output from a SNES+KSP solve.

Example:
  $ ./program -snes_monitor -ksp_monitor &> foo.txt
  $ ./sneskspplot.py foo.txt
""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('infile', metavar='NAME', help='text input file', default='foo')
parser.add_argument('--noksp', action='store_false', 
                    help='do not show KSP residuals', dest='showksp',default=True)
parser.add_argument('--showdata', action='store_true',
                    help='print raw list of norms', default=False)
parser.add_argument('--color', action='store_true',
                    help='use color in plot', default=False)
parser.add_argument('-o', help='output filename (.png,.pdf)', dest='output')

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
                if ls[k] == 'KSP' and ls[k+1] == 'Residual' and ls[k+2] == 'norm':
                    norm = float(ls[k+3])
                    data[-1].append(norm)

if args.showdata:
    print data

fig = plt.figure(figsize=(7,6))
plt.hold(True)

# actually plot points
snesmarker = 'ko'
kspmarker = 'k+'
if args.color:
    snesmarker = 'bo'
    kspmarker = 'r+'
for k in range(len(data)):
    if k == 0:
        plt.semilogy(k,data[k][0],snesmarker,markersize=9.0,label='SNES residual')
    else:
        plt.semilogy(k,data[k][0],snesmarker,markersize=9.0)
    for j in range(len(data[k])-1):
        jshift = k + (j+1) / float(len(data[k]))
        if k == 0 and j == 0:
            plt.semilogy(jshift,data[k][j+1],kspmarker,markersize=10.0,label='KSP residual')
        else:
            plt.semilogy(jshift,data[k][j+1],kspmarker,markersize=10.0)

plt.hold(False)
plt.grid(True)
plt.xlabel('SNES iteration',fontsize=18.0)
plt.xticks(range(len(data)))
plt.ylabel('residual norm',fontsize=18.0)

if args.showksp:
    plt.legend(loc='upper right')

if args.output:
    plt.savefig(args.output,bbox_inches='tight')
else:
    plt.show()

