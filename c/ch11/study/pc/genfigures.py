#!/usr/bin/env python

# from -log_view, look at time spent in:
#   SNESFunctionEval
#   SNESJacobianEval
#   PCSetUp
#   PCApply
# (latter two esp interesting form gamg vs mv vs ilu

# we see the success of gamg and mg as the number of Krylov iterations is nearly

# the issue, as much as any other, is seen in the increase in Newton iterations
# I think this is from the moving free boundary
# FIXME: calculate how many grid spaces the free boundary moves in this Halfar solution in this time interval

import numpy as np
import matplotlib.pyplot as plt

# see study.slurm for run script

# the base run
basic = "./ice -ice_verif 2 -ts_type beuler -ice_tf 10.0 -ice_dtinit 10.0 -snes_monitor -snes_converged_reason -ksp_converged_reason -ice_dtlimits"

# diffusive time-step limit function only of resolution here; note V=0 in Halfar so no CFL
dtD = { 96 : 1.76418,
        192 : 0.44521,
        384 : 0.11185,
        768 : 0.02803,
        1536 : 0.00702,
        3072 : 0.00176,
        6144 : 0.00044,
        12288 : 0.00011 }

# results is a big dictionary:
#   'name' : (options, [[mx, kspsum, snes, time], ...])
results = {
'asm-ilu' : ('-pc_type asm -sub_pc_type ilu',
   [[96, 18, 3, 16.610],
    [192, 28, 3, 17.962],
    [384, 63, 4, 19.605],
    [768, 136, 5, 21.332],
    [1536, 502, 9, 47.697],
    [3072, 1948, 15, 158.640],
    [6144, 4506, 17, 598.791],
    [12288, 12047, 18, 4445.175]]),
'asm-gamg' : ('-pc_type asm -sub_pc_type gamg',
   [[96, 15, 3, 19.903],
    [192, 20, 3, 12.866],
    [384, 38, 4, 12.982],
    [768, 60, 5, 13.500],
    [1536, 168, 9, 19.550],
    [3072, 367, 14, 64.882],
    [6144, 594, 16, 308.042],
    [12288, 876, 17, 1548.099]]),
'gamg-ns0' : ('-pc_type gamg -pc_gamg_agg_nsmooths 0',
   [[96, 12, 3, 19.852],
    [192, 12, 3, 13.332],
    [384, 18, 4, 14.756],
    [768, 28, 5, 14.580],
    [1536, 74, 9, 23.376],
    [3072, 158, 14, 63.423],
    [6144, 245, 16, 248.568],
    [12288, 366, 17, 999.116]]),
'mg' : ('-pc_type mg -pc_mg_levels LEV-2',
   [[96, 11, 3, 45.077],
    [192, 12, 3, 42.009],
    [384, 16, 4, 52.348],
    [768, 20, 5, 38.503],
    [1536, 31, 9, 34.031],
    [3072, 45, 14, 73.338],
    [6144, 50, 16, 179.758],
    [12288, 53, 17, 616.286]]),
'mg-N-1' : ('-pc_type mg -pc_mg_levels LEV-1',
   [[96, 11, 3, 42.765],
    [192, 12, 3, 34.736],
    [384, 16, 4, 59.778],
    [768, 20, 5, 82.584],
    [1536, 31, 9, 140.789],
    [3072, 45, 14, 299.176],
    [6144, 50, 16, 460.106],
    [12288, 53, 17, 1118.143]])
}

usekeys = ['asm-ilu','asm-gamg','gamg-ns0','mg']
lev = [5,6,7,8,9,10,11,12]

plt.figure(1)
for method in usekeys:
    options = results[method][0]
    data = np.array(results[method][1])
    snes = data[:,2]
    plt.semilogy(lev,snes,'k*',ms=14,label=options)
plt.grid('on')
plt.xlabel('refinement level')
plt.ylabel('Newton iterations')
plt.yticks([2, 3, 4, 5, 10, 20],['2','3','4','5','10','20'])
plt.axis([5.0,12.0,2.0,20.0])

plt.figure(2)
for method in usekeys:
    options = results[method][0]
    data = np.array(results[method][1])
    kspsum = data[:,1]
    snes = data[:,2]
    plt.semilogy(lev,kspsum / snes,'o',ms=14,label=options)
plt.grid('on')
plt.xlabel('refinement level')
plt.ylabel('Krylov iterations per Newton step')
plt.yticks([2, 3, 4, 10, 100, 1000],['2','3','4','10','100','1000'])
plt.legend(fontsize=12,loc='upper left')

plt.figure(3)
times = []
for method in usekeys:
    options = results[method][0]
    data = np.array(results[method][1])
    mx = data[4:,0]
    time = data[4:,3]
    times = times + list(time)
    m = mx * mx  # number of degrees of freedom
    plt.loglog(lev[4:],time,'*',ms=14,label=options)
    #plt.loglog(lev,(1.0e3 * time / m) / snes,'s',ms=10,label='time (ms) per DOF per Newton')
plt.axis([8.8,12.2,min(times)/2.0,2.0*max(times)])
plt.grid('on')
plt.xlabel('refinement level (dof, $\Delta x$)')
xlabeldof = []
L = 1800.000e3
for k in range(4):
    dof = '$10^{%.1f}$' % np.log10(m[k])
    dx = int(L / (3 * 2**lev[4+k]))
    xlabeldof += ['%d (%s, %d)' % (lev[4+k],dof,dx)]
print xlabeldof
plt.xticks(lev[4:],xlabeldof)
plt.ylabel('time in seconds')
plt.legend(fontsize=12,loc='upper left')

plt.show()

