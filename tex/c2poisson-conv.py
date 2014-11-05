#!/usr/bin/env python

from sys import stderr, exit, argv
from pylab import *

d = """
$ for K in 0 1 2 3 4 5 6; do ./c2poisson -da_refine $K; done
on 10 x 10 grid:  error |u-uexact|_inf = 0.000621778
on 19 x 19 grid:  error |u-uexact|_inf = 0.000155374
on 37 x 37 grid:  error |u-uexact|_inf = 3.87982e-05
on 73 x 73 grid:  error |u-uexact|_inf = 1.05331e-05
on 145 x 145 grid:  error |u-uexact|_inf = 3.17389e-06
on 289 x 289 grid:  error |u-uexact|_inf = 1.60786e-06
on 577 x 577 grid:  error |u-uexact|_inf = 1.22251e-06
"""

d2 = """
$ for K in 0 1 2 3 4 5 6; do ./c2poisson -da_refine $K -ksp_rtol 1.0e-12; done
on 10 x 10 grid:  error |u-uexact|_inf = 0.000621527
on 19 x 19 grid:  error |u-uexact|_inf = 0.000155312
on 37 x 37 grid:  error |u-uexact|_inf = 3.8823e-05
on 73 x 73 grid:  error |u-uexact|_inf = 9.71122e-06
on 145 x 145 grid:  error |u-uexact|_inf = 2.42806e-06
on 289 x 289 grid:  error |u-uexact|_inf = 6.07041e-07
on 577 x 577 grid:  error |u-uexact|_inf = 1.51761e-07
"""

N = 9 * 2**arange(7)
h = 1.0 / N

fig = figure(figsize=(7,6))

err1 = [0.000621778, 0.000155374, 3.87982e-05, 1.05331e-05, 3.17389e-06, 1.60786e-06, 1.22251e-06]
loglog(h,err1,'ks',markersize=10.0,markerfacecolor='white',markeredgewidth=1.0)
hold(True)

err2 = [0.000621527, 0.000155312, 3.8823e-05, 9.71122e-06, 2.42806e-06, 6.07041e-07, 1.51761e-07]
loglog(h,err2,'ko',markersize=10.0,markerfacecolor='white',markeredgewidth=1.0)

p = polyfit(log(h),log(err2),1)
print "result:  error decays at rate O(h^%.4f)" % p[0]
loglog(h,exp(polyval(p,log(h))),'k:')

grid(True)
axis((0.001,0.15,1.0e-7,1.0e-3))
xlabel(r'$h$',fontsize=18.0)
#xticks(array([10.0, 100.0, 1000.0, 10000.0]),('', '', '', '') )
ylabel(r'$||u - u_{ex}||_\infty$',fontsize=18.0)

#show()
savefig('c2poisson-conv.pdf',bbox_inches='tight')

