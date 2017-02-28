#!/usr/bin/env python

# from -log_view, look at time spent in:
#   SNESFunctionEval
#   SNESJacobianEval
#   PCSetUp
#   PCApply
# (latter two esp interesting form gamg vs mv vs ilu

# we see the success of gamg and mg as the number of Krylov iterations is nearly

# the issue, as much as any other, is seen in the increase in Newton iterations,
# but this is from the moving free boundary:  the Halfar solution margin moves
# 975m ~~ 1km in the 10 year run, so we expect to see Newton iteration count
# start to rise as refinement approaches 1km, and then have the count be a
# small constant plus something proportional to (1 km)/(dx km); this is about what we see

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
L = 1800.0e3

def writeit(name):
    print 'saving figure %s' % name
    plt.savefig(name,bbox_inches='tight')

plt.figure(1)
data = np.array(results['mg'][1])
mx = data[:,0]
h = L / mx
snes = data[:,2]
plt.semilogx(1000.0 / h,snes,'k*',ms=14)  # vs 975m
plt.grid('on')
plt.xlabel('h (km)')
plt.ylabel('Newton iteration count')
plt.xticks([0.04,0.1,0.2,0.5,1.0,2.0,5.0,10.0],['20','10','5','2','1','0.5','0.2','0.1'])
plt.yticks([1, 3, 5, 10, 15],['1', '3','5','10','15'])
plt.axis([0.04,10.0,0.0,20.0])
writeit('newtoniters.pdf')

plt.semilogx([0.04,0.4],[3.0,3.0],'k--',lw=2.0)
plt.semilogx([0.4,4.0],[3.0,20.0],'k--',lw=2.0)
plt.xticks([0.04,0.1,0.2,0.5,1.0,2.0,5.0,10.0],['20','10','5','2','1','0.5','0.2','0.1'])
writeit('newtonitersFIT.pdf')

plt.figure(2)
for method in usekeys:
    options = results[method][0]
    data = np.array(results[method][1])
    kspsum = data[:,1]
    snes = data[:,2]
    plt.semilogy(lev,kspsum / snes,'o',ms=12,label=options)
plt.grid('on')
plt.xlabel('refinement level')
plt.ylabel('Krylov iterations per Newton step')
plt.yticks([2, 3, 4, 10, 100, 1000],['2','3','4','10','100','1000'])
plt.legend(fontsize=12,loc='upper left')
writeit('pcksppernewton.pdf')

plt.figure(3)
times = []
for method in usekeys:
    options = results[method][0]
    data = np.array(results[method][1])
    mx = data[4:,0]
    snes = data[4:,2]
    time = data[4:,3]
    m = mx * mx  # number of degrees of freedom
    timeper = (1.0e6 * time / m) / snes
    plt.loglog(lev[4:],timeper,'o',ms=12,label=options)
plt.axis([8.8,12.2,2.0e-1,1.0e1])
plt.yticks([0.5, 1.0, 3.0, 5.0],['0.5','1','3','5'])
plt.grid('on')
xlabeldof = []
L = 1800.000e3
dof = [r'$2.4\times 10^6$', r'$9.4\times 10^6$', r'$3.8\times 10^7$', r'$1.5\times 10^8$']
for k in range(4):
    dx = int(L / (3 * 2**lev[4+k]))
    xlabeldof += ['%d\n%s\n%d m' % (lev[4+k],dof[k],dx)]
plt.xticks(lev[4:],xlabeldof)
plt.ylabel(r'wall time ($10^{-6}$ s) per d.o.f. per Newton')
plt.legend(fontsize=12,loc='upper left')
writeit('pctimeperdof.pdf')

#plt.show()

