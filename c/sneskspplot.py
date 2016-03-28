#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# FIXME add parsing so auto-generates from
#   $ ./program -snes_monitor -ksp_monitor &> foo.txt
#   $ ./sneskspplot.py foo.txt

# ./reaction -ksp_type gmres -pc_type jacobi -snes_monitor
snes = np.array([1.004378165400e+00 ,
  2.309787305521e-02,
  2.741717814657e-05 ,
  3.457837426841e-11 ])

# ./reaction -ksp_type gmres -pc_type jacobi -snes_monitor -ksp_monitor
ksp0 = np.array([4.944691520394e-01 ,
    4.000986244771e-01 ,
    2.886707778971e-01 ,
    1.680881213210e-01 ,
    7.980749093301e-02 ,
    4.605473807603e-02 ,
    2.034529122176e-02 ,
    2.872368209447e-16 ])
ksp1 = np.array([1.125600209934e-02 ,
    5.993201527867e-03 ,
    4.123570619805e-03 ,
    2.011342933375e-03 ,
    5.874400753251e-04 ,
    2.007635529883e-04 ,
    1.611630233758e-06 ,
    2.446454487517e-18 ])
ksp2 = np.array([1.333254262936e-05 ,
    7.869404871709e-06 ,
    6.334849998598e-06 ,
    4.196700444761e-06 ,
    2.575534285255e-06 ,
    7.389604225546e-07 ,
    9.558617748461e-08 ,
    5.526563734424e-21 ])

ksp = [ksp0, ksp1, ksp2]

fig = plt.figure(figsize=(7,6))

plt.semilogy(range(len(snes)),snes,'bo',markersize=10.0,label='SNES residual')
plt.hold(True)
for k in range(len(ksp)):
    jj = float(k) + (np.arange(len(ksp[k])) + 1.0) / float(len(ksp[k]))
    if k == 0:
        plt.semilogy(jj,ksp[k],'r+',markersize=10.0,label='KSP residual')
    else:
        plt.semilogy(jj,ksp[k],'r+',markersize=10.0)
plt.hold(False)

plt.grid(True)
#plt.axis((0.001,0.18,5.0,2000.0))
plt.xlabel('iteration')
plt.xticks(range(len(snes)))
plt.ylabel('residual norm',fontsize=18.0)
plt.legend(loc='upper right')

plt.show()
#filename='cg-flaw.pdf'
#print "writing %s ..." % filename
#plt.savefig(filename,bbox_inches='tight')

