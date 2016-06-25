#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def getstats(name):
    eet = np.loadtxt('../timing/unfem/' + name)
    N = len(eet)
    eet = np.reshape(eet,(N/6,6))
    h = eet[:,0]
    err = eet[:,1]
    evals = np.array(eet[:,2],dtype=int)
    readmeshstagetime = eet[:,3]
    setupstagetime = eet[:,4]
    solverstagetime = eet[:,5]
    return h, err, evals, readmeshstagetime, setupstagetime, solverstagetime

def saveit(outname):
    print "writing %s ..." % outname
    plt.savefig(outname,bbox_inches='tight')
    plt.clf()

h, err, evals, _, _, _ = getstats('snes-fd-ErrorsEvalsTimes')

# figure for -snes_fd run with both errors and evals
_, ax1 = plt.subplots()
ax1.loglog(h,err,'ko',markersize=16.0,markeredgewidth=2.0,markerfacecolor='w')
ax1.grid(True)
ax1.set_xlabel(r'$h$',fontsize=20.0)
ax1.set_ylabel(r'$\|u-u_{ex}\|_\infty$  (circles)',fontsize=18.0)
ax2 = ax1.twinx()
ax2.loglog(h,evals,'k*',markersize=14.0,markeredgewidth=2.0)
ax2.set_ylabel('FormFunction() evaluations  (stars)',fontsize=16.0)
ax1.set_xlim(3.0e-2,3.0)
#plt.show()
saveit('unfem-snesfd.pdf')

# figure for case 0,1,2 errors
h0, err0, _, rmst, sust, sost = getstats('ErrorsEvalsTimes')
p = np.polyfit(np.log(h0),np.log(err0),1)
print 'convergence for case 0 at rate h^%.3f' % p[0]
plt.loglog(h0,err0,'ko',markersize=9.0,markerfacecolor='k',
           label=r'case 0  $O(h^{%.3f})$' % p[0])
plt.hold(True)
h, err, _, _, _, _ = getstats('case1-ErrorsEvalsTimes')
p = np.polyfit(np.log(h),np.log(err),1)
print 'convergence for case 1 at rate h^%.3f' % p[0]
plt.loglog(h,err,'ks',markersize=14.0,markerfacecolor='w',alpha=1.0,
           label=r'case 1  $O(h^{%.3f})$' % p[0])
h, err, _, _, _, _ = getstats('case2-ErrorsEvalsTimes')
p = np.polyfit(np.log(h[1:]),np.log(err[1:]),1)
print 'convergence for case 2 at rate h^%.3f' % p[0]
plt.loglog(h,err,'kd',markersize=14.0,markerfacecolor='w',alpha=1.0,
           label=r'case 2  $O(h^{%.3f})$' % p[0])
plt.loglog(h0,err0,'ko',markersize=9.0,markerfacecolor='k')
plt.hold(False)
plt.grid(True)
plt.legend(loc='upper left')
plt.xlabel(r'$h$',fontsize=20.0)
plt.ylabel(r'$\|u-u_{ex}\|_\infty$',fontsize=18.0)
plt.xlim(2.0e-3,3.0)
#plt.show()
saveit('unfem-error.pdf')

plt.loglog(h,rmst,'kx',markersize=12.0,markeredgewidth=2.0,label='read mesh')
plt.hold(True)
plt.loglog(h,sust,'k+',markersize=14.0,markeredgewidth=2.0,label='set-up')
plt.loglog(h,sost,'k*',markersize=14.0,markeredgewidth=2.0,label='solver')
plt.grid(True)
plt.xlabel(r'$h$',fontsize=20.0)
plt.ylabel('time for stage  (seconds)',fontsize=18.0)
plt.xlim(2.0e-3,3.0)
plt.hold(False)
plt.legend()
#plt.show()
saveit('unfem-times.pdf')

