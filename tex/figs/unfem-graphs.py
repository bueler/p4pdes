#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def getheet(name):
    eet = np.loadtxt('../timing/unfem/' + name)
    N = len(eet)
    eet = np.reshape(eet,(N/4,4))
    h = eet[:,0]
    err = eet[:,1]
    evals = np.array(eet[:,2],dtype=int)
    time = eet[:,3]
    return h, err, evals, time

def saveit(outname):
    print "writing %s ..." % outname
    plt.savefig(outname,bbox_inches='tight')
    plt.clf()

h, err, evals, time = getheet('snes-fd-ErrorsEvalsTimes')

fig, ax1 = plt.subplots()
ax1.loglog(h,err,'ko',markersize=16.0,markerfacecolor='w')
ax1.grid(True)
ax1.set_xlabel(r'$h_{max}$',fontsize=20.0)
ax1.set_ylabel(r'$\|u-u_{ex}\|_\infty$  (circles)',fontsize=18.0)
ax2 = ax1.twinx()
ax2.loglog(h,evals,'k*',markersize=14.0)
ax2.set_ylabel('FormFunction() evaluations  (stars)',fontsize=16.0)
ax1.set_xlim(3.0e-2,3.0)
#plt.show()
saveit('unfem-snesfd.pdf')

snesfderr = err.copy()
h, err, evals, time = getheet('ErrorsEvalsTimes')
p = np.polyfit(np.log(h),np.log(err),1)
print 'convergence at rate h^%.3f' % p[0]

fig = plt.figure(1)
plt.loglog(h[:5],snesfderr,'ko',markersize=16.0,markerfacecolor='w')
plt.grid(True)
plt.hold(True)
plt.loglog(h,err,'ko',markersize=10.0,markerfacecolor='k')
plt.loglog(h,np.exp(p[0]*np.log(h)+p[1]),'k--')
plt.text(1.8*h[3],1.1*err[3],r'$O(h_{max}^{%.3f})$' % p[0],fontsize=18.0)
plt.xlabel(r'$h_{max}$',fontsize=20.0)
plt.ylabel(r'$\|u-u_{ex}\|_\infty$',fontsize=18.0)
plt.xlim(2.0e-3,3.0)
plt.hold(False)
#plt.show()
saveit('unfem-error.pdf')

#FIXME: figure with stage times: read mesh, assemble, solve

