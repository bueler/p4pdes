#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

datasource='''
RUNS TO SHOW CG ITERATIONS INCREASE WITH GRID REFINEMENT:
$ for NN in 1 2 3 4 5; do
> ./poisson -da_refine $NN -ksp_type cg -pc_type none -ksp_converged_reason; done
AND
$ for NN in 1 2 3 4 5; do
> ./poisson -da_refine $NN -ksp_type cg -pc_type icc -ksp_converged_reason; done
'''

N = np.array([4, 5, 6, 7, 8])
h = 1.0 / 2**N
pcnone = [36, 73, 148, 299, 606]
pcicc  = [12, 23, 44, 88, 177]
ohinv = 1.0 / h

fig = plt.figure(figsize=(7,6))

plt.loglog(h,pcnone,'k*',markersize=14.0,label=r'-pc_type none')
plt.hold(True)
plt.loglog(h,pcicc,'ko',markersize=10.0,label=r'-pc_type icc')
plt.loglog(h,ohinv,'k--',lw=2.0)
plt.text(0.015,70,r'$O(h^{-1})$',fontsize=20.0)
plt.hold(False)

plt.grid(True)
plt.axis((0.003,0.1,10,1000))
plt.xlabel(r'$h$',fontsize=20.0)
plt.xticks(np.array([0.003, 0.01, 0.03, 0.1]),('.003', '.01', '.03', '.1'),fontsize=14.0)
plt.ylabel('CG iterations',fontsize=16.0)
plt.yticks(np.array([10.0, 100.0, 1000.0]),('10', '100', '1000'),fontsize=14.0)

plt.legend()

#plt.show()
filename='poisson-cg-scale.pdf'
print "writing %s ..." % filename
plt.savefig(filename,bbox_inches='tight')

