#!/usr/bin/env python

"""
Generate residual norm figure from PETSc monitor output from a SNES+KSP solve.
Run with -h for help.
"""

import numpy as np
import matplotlib.pyplot as plt

def getskoptions():
    """Returns args structure.  Flags are args.color, args.showksp, args.showsnes, args.ksptrue, args.showdata."""

    import argparse
    parser = argparse.ArgumentParser(description=
    """
    Generate residual norm figure from PETSc monitor output from a SNES+KSP run.
    Or show SNES or KSP norms only.  Can show true residual norms from KSP.

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
                        help='output filename (uses matplotlib.pyplot.savefig)')
    parser.add_argument('--showdata', action='store_true',
                        help='print raw list of norms', default=False)
    return parser.parse_args()

def readskfile(infile, showksp=True, ksptrue=False):
    snesdata = []
    kspdata = []
    snescounter = -1
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
                    snescounter += 1
                if showksp:
                    if ksptrue:
                        if ls[k] == 'true' and ls[k+1] == 'resid' and ls[k+2] == 'norm':
                            norm = float(ls[k+3])
                            kspdata.append([snescounter, norm])
                    else:
                        if ls[k] == 'KSP' and ls[k+1] == 'Residual' and ls[k+2] == 'norm':
                            norm = float(ls[k+3])
                            kspdata.append([snescounter, norm])
    return snesdata, kspdata

def getkspruns(kspdata):
    count = -1
    kspruns = []
    for k in range(len(kspdata)):
        if count < 0 or count != kspdata[k][0]:
            # start a new run
            kspruns.append([kspdata[k][1]])
            count = kspdata[k][0]
        else: # continue run
            kspruns[-1].append(kspdata[k][1])
    return kspruns

def genskfigure(snesdata, kspdata,
                docolor=False, showsnes=True, showksp=True, ksptrue=False, msize=10.0):
    """Generate semilogy() figure with SNES and KSP residual norms."""

    fig = plt.figure(figsize=(7,6))
    ax = fig.gca()

    # determine symbols & labels
    if docolor:
        mcolor = 'b'
    else:
        mcolor = 'k'
    snesmarker = mcolor + 'o'
    kspmarker = mcolor + '*'
    if not showsnes:
        kspmarker = snesmarker
    if ksptrue:
        ksplabel = 'KSP true residual norm'
    else:
        ksplabel = 'KSP residual norm'

    # plot points
    if showsnes:
        assert len(snesdata) > 0, 'empty snesdata'
        for k in range(len(snesdata)):
            if k == 0:
                plt.semilogy(0,snesdata[0],snesmarker,markersize=msize,
                             label='SNES residual norm')
            else:
                plt.semilogy(k,snesdata[k],snesmarker,markersize=msize)
        plt.xlabel('SNES iteration',fontsize=18.0)
    if showksp:
        assert len(kspdata) > 0, 'empty kspdata'
        kspruns = getkspruns(kspdata)
        for j in range(len(kspruns)):
            # variable number of KSP markers to show between SNES markers
            nksp = len(kspruns[j])
            if nksp > 0:
                jj = np.arange(nksp)
                jjshift = j + jj / float(nksp)
                if j == 0:
                    plt.semilogy(jjshift,kspruns[j],kspmarker+'-',markersize=msize-2,label=ksplabel)
                else:
                    plt.semilogy(jjshift,kspruns[j],kspmarker+'-',markersize=msize-2)

    # default decorations
    plt.ylabel('residual norm',fontsize=18.0)
    plt.grid(True)
    if showsnes:
        plt.xticks(range(len(snesdata)))
        if showksp:
            plt.legend(loc='upper right')
    else:
        plt.xticks(range(len(kspdata)))

    return fig, ax

if __name__=='__main__':
    args = getskoptions()
    snesdata, kspdata = readskfile(open(args.infile, 'r'),
                                   showksp=args.showksp, ksptrue=args.ksptrue)
    if args.showdata:
        print 'SNES residual norms:'
        print snesdata
        print 'KSP residual norms ([count, norm]):'
        print kspdata
    genskfigure(snesdata, kspdata,
                docolor=args.color, showsnes=args.showsnes, showksp=args.showksp,
                ksptrue=args.ksptrue, msize=args.msize)
    if args.output:
        plt.savefig(args.output,bbox_inches='tight')
    elif args.showit:
        plt.show()

