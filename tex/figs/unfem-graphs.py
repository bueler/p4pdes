#!/usr/bin/env python

# uses data files from  tex/timing/unfem/

import numpy as np
import matplotlib.pyplot as plt

# returns:
#     N = number of runs
#     JEpresent = True if "Jacobian eval" present
#     h = array of N maximum triangle side lengths
#     err = array of N numerical errors (inf norm)
#     evals = integer array of N residual-fcn evaluations (SNESFunctionEval)
#     solver = array of N "Solver" stage times
#     ost = Nx5 array of OTHER stage times; each line is
#         Read mesh, Set-up, Residual eval[, Jacobian eval]
def getstats(name):
    eet = np.loadtxt('../timing/unfem/' + name)
    N = np.shape(eet)[0]
    JEpresent = (np.shape(eet)[1] == 8)
    h = eet[:,0]
    err = eet[:,1]
    evals = np.array(eet[:,2],dtype=int)
    solver = eet[:,5]
    if JEpresent:
        ost = eet[:,(3,4,6,7)]
    else:
        ost = eet[:,(3,4,6)]
    return N, JEpresent, h, err, evals, solver, ost

def saveit(outname):
    print "writing %s ..." % outname
    plt.savefig(outname,bbox_inches='tight')
    plt.clf()

_, _, hfd, errfd, evalsfd, _, _ = getstats('snes-fd-case0')
_, _, hmf, errmf, evalsmf, _, _ = getstats('snes-mf-case0')

# figure for -snes_fd, -snes_mf runs with both errors and evals
_, ax1 = plt.subplots()
ax1.loglog(hmf,errmf,'ko',markersize=16.0,markeredgewidth=2.0,markerfacecolor='w')
#on top: ax1.loglog(hfd,errfd,'ks',markersize=12.0,markeredgewidth=2.0,markerfacecolor='w')
ax1.grid(True)
ax1.set_xlabel(r'$h$',fontsize=20.0)
ax1.set_ylabel(r'$\|u-u_{ex}\|_\infty$  (open circles)',fontsize=18.0)
ax2 = ax1.twinx()
ax2.loglog(hfd,evalsfd,'k*',markersize=16.0,markeredgewidth=2.0,label=r'-snes_fd')
ax2.hold(True)
ax2.loglog(hmf,evalsmf,'ks',markersize=12.0,markeredgewidth=2.0,label=r'-snes_mf')
ax2.set_ylabel('FormFunction() evaluations  (solid)',fontsize=16.0)
ax2.hold(False)
ax2.legend(loc='lower center')
ax1.set_xlim(1.0e-2,3.0)
saveit('unfem-fdmf.pdf')

_, _, h0, err0, _, _, _ = getstats('case0')
_, _, h1, err1, _, sost1, ost1 = getstats('case1')
ost1 = ost1.sum(1)
_, _, h2, err2, _, _, _ = getstats('case2')

# figure for case 0,1,2 errors
p = np.polyfit(np.log(h0[2:-1]),np.log(err0[2:-1]),1)
print 'convergence for case 0 at rate h^%.3f' % p[0]
plt.loglog(h0,err0,'ko',markersize=9.0,markerfacecolor='k',
           label=r'case 0  $O(h^{%.3f})$' % p[0])
plt.hold(True)
p = np.polyfit(np.log(h1[2:-1]),np.log(err1[2:-1]),1)
print 'convergence for case 1 at rate h^%.3f' % p[0]
plt.loglog(h1,err1,'ks',markersize=14.0,markerfacecolor='w',alpha=1.0,
           label=r'case 1  $O(h^{%.3f})$' % p[0])
p = np.polyfit(np.log(h2[2:-1]),np.log(err2[2:-1]),1)
print 'convergence for case 2 at rate h^%.3f' % p[0]
plt.loglog(h2,err2,'kd',markersize=14.0,markerfacecolor='w',alpha=1.0,
           label=r'case 2  $O(h^{%.3f})$' % p[0])
plt.loglog(h0,err0,'ko',markersize=9.0,markerfacecolor='k')
plt.hold(False)
plt.grid(True)
plt.legend(loc='upper left')
plt.xlabel(r'$h$',fontsize=20.0)
plt.ylabel(r'$\|u-u_{ex}\|_\infty$',fontsize=18.0)
plt.xlim(2.0e-3,3.0)
saveit('unfem-error.pdf')

# figure for case 1 stage timing
plt.loglog(h1,sost1,'k*',markersize=14.0,markeredgewidth=2.0,label='solver')
plt.hold(True)
plt.loglog(h1,ost1,'kx',markersize=12.0,markeredgewidth=2.0,label='other stages')
plt.grid(True)
plt.xlabel(r'$h$',fontsize=20.0)
plt.ylabel('time for stage  (seconds)',fontsize=18.0)
plt.xlim(2.0e-3,3.0)
plt.hold(False)
plt.legend()
saveit('unfem-times.pdf')

