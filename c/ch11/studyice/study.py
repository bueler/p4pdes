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

# 'name' : (options, [[mx, kspsum, snes, time], ...])
data = {
'asmilu' : ('-pc_type asm -sub_pc_type ilu',
   [[96, 18, 3, 16.610],
    [192, 28, 3, 17.962],
    [384, 63, 4, 19.605],
    [768, 136, 5, 21.332],
    [1536, 502, 9, 47.697],
    [3072, 1948, 15, 158.640],
    [6144, 4506, 17, 598.791],
    [12288, 12047, 18, 4445.175]]),
'asmgamg' : ('-pc_type asm -sub_pc_type gamg',
   [[96, 15, 3, 19.903],
    [192, 20, 3, 12.866],
    [384, 38, 4, 12.982],
    [768, 60, 5, 13.500],
    [1536, 168, 9, 19.550],
    [3072, 367, 14, 64.882],
    [6144, 594, 16, 308.042],
    [12288, 876, 17, 1548.099]]),

}

# FIXME from here
mx = data[:,0]
kspsum = data[:,1]
snes = data[:,2]
time = data[:,3]

m = mx * mx  # number of degrees of freedom

plt.loglog(m,snes,'o',ms=14,label='Newton iterations')
plt.loglog(m,kspsum / snes,'*',ms=14,label='KSP iterations per Newton')
plt.loglog(m,(1.0e3 * time / m) / snes,'s',ms=10,label='time (ms) per DOF per Newton')
plt.loglog(m,time,'d',ms=10,label='total time')

plt.grid('on')
plt.xlabel('Degrees Of Freedom')
plt.legend(fontsize=12,loc='upper left')

plt.show()

