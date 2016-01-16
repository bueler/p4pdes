#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

objdata = """
$ for LEV in 0 1 2 3; do ./plap -snes_fd_color -snes_fd_function -snes_converged_reason -da_refine $LEV; done
grid of 3 x 3 = 9 interior nodes (element dims 0.25x0.25)
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 6
numerical error:  |u-u_exact|/|u_exact| = 8.081e-03
grid of 5 x 5 = 25 interior nodes (element dims 0.166667x0.166667)
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 7
numerical error:  |u-u_exact|/|u_exact| = 2.929e-03
grid of 9 x 9 = 81 interior nodes (element dims 0.1x0.1)
Nonlinear solve did not converge due to DIVERGED_LINEAR_SOLVE iterations 8
numerical error:  |u-u_exact|/|u_exact| = 9.253e-04
grid of 17 x 17 = 289 interior nodes (element dims 0.0555556x0.0555556)
Nonlinear solve did not converge due to DIVERGED_LINE_SEARCH iterations 20
numerical error:  |u-u_exact|/|u_exact| = 2.582e-04
"""

quad1data = """
$ for LEV in 0 1 2 3 4 5 6 7; do ./plap -snes_fd_color -snes_converged_reason -da_refine $LEV -plap_quaddegree 1; done
grid of 3 x 3 = 9 interior nodes (element dims 0.25x0.25)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
numerical error:  |u-u_exact|/|u_exact| = 1.589e-02
grid of 5 x 5 = 25 interior nodes (element dims 0.166667x0.166667)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
numerical error:  |u-u_exact|/|u_exact| = 5.491e-03
grid of 9 x 9 = 81 interior nodes (element dims 0.1x0.1)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
numerical error:  |u-u_exact|/|u_exact| = 1.667e-03
grid of 17 x 17 = 289 interior nodes (element dims 0.0555556x0.0555556)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
numerical error:  |u-u_exact|/|u_exact| = 4.600e-04
grid of 33 x 33 = 1089 interior nodes (element dims 0.0294118x0.0294118)
Nonlinear solve converged due to CONVERGED_SNORM_RELATIVE iterations 6
numerical error:  |u-u_exact|/|u_exact| = 1.219e-04
grid of 65 x 65 = 4225 interior nodes (element dims 0.0151515x0.0151515)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 9
numerical error:  |u-u_exact|/|u_exact| = 3.138e-05
grid of 129 x 129 = 16641 interior nodes (element dims 0.00769231x0.00769231)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 11
numerical error:  |u-u_exact|/|u_exact| = 7.964e-06
grid of 257 x 257 = 66049 interior nodes (element dims 0.00387597x0.00387597)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 12
numerical error:  |u-u_exact|/|u_exact| = 2.011e-06
"""

p4data = """
$ for LEV in 0 1 2 3 4 5 6 7 8; do ./plap -snes_fd_color -snes_converged_reason -da_refine $LEV; done
grid of 3 x 3 = 9 interior nodes (element dims 0.25x0.25)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
numerical error:  |u-u_exact|/|u_exact| = 8.081e-03
grid of 5 x 5 = 25 interior nodes (element dims 0.166667x0.166667)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
numerical error:  |u-u_exact|/|u_exact| = 2.929e-03
grid of 9 x 9 = 81 interior nodes (element dims 0.1x0.1)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
numerical error:  |u-u_exact|/|u_exact| = 9.253e-04
grid of 17 x 17 = 289 interior nodes (element dims 0.0555556x0.0555556)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
numerical error:  |u-u_exact|/|u_exact| = 2.582e-04
grid of 33 x 33 = 1089 interior nodes (element dims 0.0294118x0.0294118)
Nonlinear solve converged due to CONVERGED_SNORM_RELATIVE iterations 6
numerical error:  |u-u_exact|/|u_exact| = 6.863e-05
grid of 65 x 65 = 4225 interior nodes (element dims 0.0151515x0.0151515)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 9
numerical error:  |u-u_exact|/|u_exact| = 1.769e-05
grid of 129 x 129 = 16641 interior nodes (element dims 0.00769231x0.00769231)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 11
numerical error:  |u-u_exact|/|u_exact| = 4.490e-06
grid of 257 x 257 = 66049 interior nodes (element dims 0.00387597x0.00387597)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 12
numerical error:  |u-u_exact|/|u_exact| = 1.135e-06
grid of 513 x 513 = 263169 interior nodes (element dims 0.00194553x0.00194553)
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 14
numerical error:  |u-u_exact|/|u_exact| = 2.840e-07
"""

mx = np.array([3, 5, 9, 17,
               33, 65, 129, 257, 513])
h = 1.0 / (mx+1.0)
#objp4err = [8.081e-03, 2.929e-03, 9.253e-04, 2.582e-04]   # same for first four
quad1err = [1.589e-02, 5.491e-03, 1.667e-03, 4.600e-04,
            1.219e-04, 3.138e-05, 7.964e-06, 2.011e-06]
p4err = [8.081e-03, 2.929e-03, 9.253e-04, 2.582e-04,
       6.863e-05, 1.769e-05, 4.490e-06, 1.135e-06, 2.840e-07]

p = np.polyfit(np.log(h),np.log(p4err),1)
print "result:  error decays at rate O(h^%.4f)" % p[0]

fig = plt.figure(figsize=(7,6))
plt.hold(True)

plt.loglog(h[0:4],p4err[0:4],'ko',markersize=16.0,markerfacecolor='w')
plt.loglog(h[0:8],quad1err,'ks',markersize=10.0,markerfacecolor='w')
plt.loglog(h,p4err,'ko',markersize=10.0)
plt.loglog(h,np.exp(np.polyval(p,np.log(h))),'k--',lw=2.0)

plt.text(0.012,4.0e-6,r'$O(h^{%.4f})$' % p[0],fontsize=20.0)

plt.grid(True)
plt.axis((1.0e-3,0.35,1.0e-7,3.0e-2))
plt.xlabel(r'$h_x =h_y$',fontsize=20.0)
plt.xticks(np.array([0.001, 0.01, 0.1]),
                    (r'$0.001$',r'$0.01$',r'$0.1$'),
                    fontsize=16.0)
plt.ylabel(r'$|u - u_{ex}|_\infty / |u|_\infty$',fontsize=18.0)
plt.yticks(np.array([1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]),
                    (r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'),
                    fontsize=16.0)

#plt.show()
filename='plap-conv.pdf'
print "writing %s ..." % filename
plt.savefig(filename,bbox_inches='tight')

