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

parser.add_argument('infile', metavar='NAME', help='text input file', default='foo.txt')
parser.add_argument('--color', action='store_true',
                    help='use color in plot', default=False)
parser.add_argument('--ksptrue', action='store_true',
                    help='read output from -ksp_monitor_true_residual', default=False)
parser.add_argument('--markersize', dest='msize',
                    help='use this markersize for major symbols', default=10.0)
parser.add_argument('--noksp', action='store_false', 
                    help='do not show KSP residuals', dest='showksp',default=True)
parser.add_argument('--noshow', action='store_false',
                    help='do not show figure', dest='showit',default=True)
parser.add_argument('--nosnes', action='store_false',
                    help='do not show SNES residuals', dest='showsnes',default=True)
parser.add_argument('-o', dest='output',
                    help='output filename (using matplotlib.pyplot.savefig; e.g. .png,.pdf)')
parser.add_argument('--showdata', action='store_true',
                    help='print raw list of norms', default=False)
args = parser.parse_args()

# note: flags are args.color, args.showksp, args.showsnes, args.ksptrue, args.showdata

infile = open(args.infile, 'r')

snesdata = []
snescount = -1
kspdata = []
for line in infile:
    # strip comment lines starting with "#"
    line = line.partition('#')[0]
    line = line.rstrip()
    if line: # if content remains
        ls = line.split()
        for k in range(len(ls)):
            if ls[k] == 'SNES' and ls[k+1] == 'Function' and ls[k+2] == 'norm':
                norm = float(ls[k+3])
                snesdata.append(norm)
                snescount += 1
            if args.showksp:
                if args.ksptrue:
                    if ls[k] == 'true' and ls[k+1] == 'resid' and ls[k+2] == 'norm':
                        norm = float(ls[k+3])
                        kspdata.append([snescount, norm])
                else:
                    if ls[k] == 'KSP' and ls[k+1] == 'Residual' and ls[k+2] == 'norm':
                        norm = float(ls[k+3])
                        kspdata.append([snescount, norm])

if args.showdata:
    print 'SNES residual norms:'
    print snesdata
    print 'KSP residual norms ([count, norm]):'
    print kspdata

fig = plt.figure(figsize=(7,6))

# determine symbols & labels
if args.color:
    mcolor = 'b'
else:
    mcolor = 'k'
snesmarker = mcolor + 'o'
kspmarker = mcolor + '*'
if not args.showsnes:
    kspmarker = snesmarker
if args.ksptrue:
    ksplabel = 'KSP true residual norm'
else:
    ksplabel = 'KSP residual norm'

# plot points
if args.showsnes and snescount >= 0:
    for k in range(len(snesdata)):
        if k == 0:
            plt.semilogy(0,snesdata[0],snesmarker,markersize=args.msize,
                         label='SNES residual norm')
        else:
            plt.semilogy(k,snesdata[k],snesmarker,markersize=args.msize)
    plt.xlabel('SNES iteration',fontsize=18.0)

if args.showksp and len(kspdata) > 0:
    count = -1
    kspruns = []
    for k in range(len(kspdata)):
        if kspdata[k][0] > 0 and count == kspdata[k][0]:
            kspruns[-1].append(kspdata[k][1])
        else:
            kspruns.append([kspdata[k][1]])
            count = kspdata[k][0]
    for j in range(len(kspruns)):
        # variable number of KSP markers to show between SNES markers
        nksp = len(kspruns[j])
        if nksp > 0:
            jj = np.arange(nksp)
            jjshift = j + jj / float(nksp)
            if j == 0:
                plt.semilogy(jjshift,kspruns[j],kspmarker+'-',markersize=args.msize-2,label=ksplabel)
            else:
                plt.semilogy(jjshift,kspruns[j],kspmarker+'-',markersize=args.msize-2)

plt.grid(True)
if args.showsnes:
    plt.xticks(range(len(snesdata)))
else:
    plt.xticks(range(len(kspdata)))

if args.showsnes and args.showksp:
    plt.legend(loc='upper right')

if args.output:
    plt.savefig(args.output,bbox_inches='tight')
elif args.showit:
    plt.show()

