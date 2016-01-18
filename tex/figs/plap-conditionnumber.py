#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

mx = np.array([3, 5, 9, 17, 33, 65, 129])
h = 1.0 / (mx+1.0)

# with command line
# for LEV in 0 1 2 3 4 5 6; do
#>   ./plap -pc_type none -ksp_type cg -ksp_rtol 1.0e-14 -da_refine $LEV \
#>      -ksp_compute_singularvalues; done
condA = [10.7515, 29.2194, 95.9797, 361.692, 1451.27, 5947.31, 24459.6]

# with added option -plap_alpha 0.1, for LEV in 0 1 2 3 4 only; beyond that get
# DIVERGED_LINEAR_SOLVE, but of course this is unpreconditioned
badN = 5
condAbad = [64.2929, 641.658, 6409.84, 52084.1, 342843.0]


filename='plap-conditionnumber.pdf'
fig = plt.figure(figsize=(7,6))
plt.hold(True)

plt.loglog(h,condA,'ko',markersize=10.0)
p = np.polyfit(np.log(h),np.log(condA),1)
print "result with alpha=1:  condition number grows at rate O(h^%.4f)" % p[0]
plt.loglog(h,np.exp(np.polyval(p,np.log(h))),'k--',lw=2.0)
plt.text(0.05,2.0e1,r'$O(h^{%.4f})$' % p[0],fontsize=20.0)

plt.loglog(h[0:badN],condAbad,'ks',markersize=8.0)
pbad = np.polyfit(np.log(h[0:badN]),np.log(condAbad),1)
print "result with alpha=0.1:  condition number grows at rate O(h^%.4f)" % pbad[0]
plt.loglog(h[0:badN],np.exp(np.polyval(pbad,np.log(h[0:badN]))),'k--',lw=2.0)
plt.text(0.07,3.0e4,r'$O(h^{%.4f})$' % pbad[0],fontsize=20.0)

plt.grid(True)
plt.axis((0.006,0.4,6.0,7.0e5))
plt.xlabel(r'$h=h_x =h_y$',fontsize=20.0)
plt.xticks(np.array([0.01, 0.03, 0.1, 0.3]),
                    (r'$0.01$',r'$0.03$',r'$0.1$',r'$0.3$'),
                    fontsize=16.0)
plt.ylabel('condition number',fontsize=18.0)
#plt.yticks(np.array([1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]),
#                    (r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'),
#                    fontsize=16.0)

#plt.show()
print "writing %s ..." % filename
plt.savefig(filename,bbox_inches='tight')

