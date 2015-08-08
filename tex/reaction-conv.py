#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

d = """
$ for N in 0 2 4 6 8 10 12 14; do
>   ./reaction -da_refine $N -snes_rtol 1.0e-10;
> done
on 9 point grid:  |u-u_exact|_inf/|u|_inf = 0.000188753
on 33 point grid:  |u-u_exact|_inf/|u|_inf = 1.1825e-05
on 129 point grid:  |u-u_exact|_inf/|u|_inf = 7.39662e-07
on 513 point grid:  |u-u_exact|_inf/|u|_inf = 4.62255e-08
on 2049 point grid:  |u-u_exact|_inf/|u|_inf = 2.88941e-09
on 8193 point grid:  |u-u_exact|_inf/|u|_inf = 1.75603e-10
on 32769 point grid:  |u-u_exact|_inf/|u|_inf = 6.77346e-12
on 131073 point grid:  |u-u_exact|_inf/|u|_inf = 7.05476e-13
"""


N = 8 * 4**np.arange(8)
h = 1.0 / N
err = [0.000188753, 1.1825e-05, 7.39662e-07, 4.62255e-08, 2.88941e-09, 1.75603e-10, 6.77346e-12, 7.05476e-13]

p = np.polyfit(np.log(h),np.log(err),1)
print "result:  error decays at rate O(h^%.4f)" % p[0]

fig = plt.figure(figsize=(7,6))

plt.loglog(h,err,'ko',markersize=10.0)
plt.hold(True)
plt.loglog(h,np.exp(np.polyval(p,np.log(h))),'k--',lw=2.0)
plt.text(0.005,7.0e-8,r'$O(h^{%.4f})$' % p[0],fontsize=20.0)
plt.hold(False)

plt.grid(True)
plt.axis((3.0e-6,0.3,1.0e-13,1.0e-3))
plt.xlabel(r'$h$',fontsize=20.0)
plt.xticks(np.array([1.0e-5, 0.0001, 0.001, 0.01, 0.1]),(r'$10^{-5}$', r'$10^{-4}$', '.001', '.01', '.1'),fontsize=14.0)
plt.ylabel(r'$||u - u_{ex}||_\infty / ||u||_\infty$',fontsize=18.0)

#plt.show()
filename='reaction-conv.pdf'
print "writing %s ..." % filename
plt.savefig(filename,bbox_inches='tight')

