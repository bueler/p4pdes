#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# see fiftyka.slurm for run script

# the base run:  defaults: -ts_type arkimex -ts_arkimex_type 3 -snes_type vinewtonrsls -ksp_type gmres
basic = "./ice -da_refine $LEV -pc_type mg -pc_mg_levels $(( $LEV - 2 )) -ice_dtlimits -ice_tf 50000 -ice_dtinit 1.0 -ts_max_snes_failures -1 -ts_adapt_scale_solve_failed 0.9 -ice_dump -snes_converged_reason"

lev = [6,7,8,9]
L = 1800.0e3
dur = 50000.0

# mx, steps, dtD, dtCFL
raw = [[192,  1496, 2.667,   46.88],
       [384,  1924, 0.6866,  23.44],
       [768,  2167, 0.1718,  11.72],
       [1536, 2517, 0.04297, 5.859]]
data = np.array(raw)

mx = data[:,0]
m = mx * mx
h = L / mx
dtactual = dur / data[:,1]
dtFE = data[:,2]
dtCFL = data[:,3]

def writeit(fname):
    print 'saving figure %s' % fname
    plt.savefig(fname,bbox_inches='tight')

plt.figure(1)
plt.semilogy(lev,dtactual,'o',ms=14,label=r'average $\Delta t$')
plt.semilogy(lev,dtCFL,'*',ms=14,label=r'$\Delta t_{CFL}$')
plt.semilogy(lev,dtFE,'s',ms=14,label=r'$\Delta t_{FE}$')
plt.grid('on')
xticlabel = []
for k in range(4):
    xticlabel.append('%.1f' % (h[k] / 1000.0))
plt.xticks(lev,xticlabel)
plt.xlabel('h  (km)')
plt.ylabel('time step (a)')
plt.axis([lev[0]-0.2,lev[-1]+0.2,1.0e-2,1.0e2])
plt.legend(fontsize=14,loc='lower left')
writeit('dtfiftyka.pdf')

#plt.show()

