#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# see weakstudy.slurm for run script and slurm.10883 for results

# the base run
basic = "./ice -ice_verif 2 -ts_type beuler -ice_tf 10.0 -ice_dtinit 10.0 -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -pc_mg_levels $(( $LEV - 1 )) -da_refine $LEV"

# on mx=1536:
#   max D_SIA 1.550 m2s-1
#   dtD 7.017e-03 a   <-- time step limit for Forward Euler

dtratios = 10.0 / (0.007017 / np.array([1.0,2.0,4.0,8.0])**2)

# mx, rank, Newtons, time
raw  = [[1536,  1,  9,  146.907], #2m16.907s
        [3072,  4,  14, 339.587], #5m39.587s
        [6144,  16, 16, 630.244], #10m30.244s
        [12288, 64, 17, 960.936]] #16m0.936s
data = np.array(raw)

lev = [9,10,11,12]
L = 1800.0e3

mx = data[:,0]
m = mx * mx
h = L / mx
rank = data[:,1]
snes = data[:,2]
time = data[:,3]

def writeit(fname):
    print 'saving figure %s' % fname
    plt.savefig(fname,bbox_inches='tight')

plt.figure(1)
plt.plot(lev,time / snes,'o',ms=12)
plt.grid('on')
dof = [r'$2.4\times 10^6$', r'$9.4\times 10^6$', r'$3.8\times 10^7$', r'$1.5\times 10^8$']
xlabeldof = []
for k in range(4):
    xlabeldof += ['%d\n%s\n%d m' % (rank[k],dof[k],h[k])]
plt.xticks(lev,xlabeldof)
plt.ylabel('time (s) per Newton step')
plt.axis([8.8,12.2,10.0,70.0])
plt.yticks([20, 40, 60],['20','40','60'])
writeit('timepernewtonweak.pdf')

plt.text(lev[0]+0.1,60.0,r'the good news: $\frac{\Delta t}{\Delta t_{FE}}$',color='r',fontsize=30)
for k in range(4):
    plt.text(lev[k]-0.2,30.0,'%.1e' % dtratios[k],color='r',fontsize=12)
writeit('withdtratioweak.pdf')

#plt.show()

