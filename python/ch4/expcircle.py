#!/usr/bin/env python3
'''Newton's method for a two-variable system.  No analytical Jacobian.
Run with -snes_fd (default) or -snes_mf.'''

import sys
from numpy import exp
import petsc4py
petsc4py.init(sys.argv)  # must come before "import PETSc"
from petsc4py import PETSc

def formFunction(_snes, X, F):
    '''Evaluate F(X).'''
    b = 2.0
    ax = X.getArray(readonly=1)
    af = F.getArray(readonly=0)
    af[0] = (1.0 / b) * exp(b * ax[0]) - ax[1]
    af[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0

x = PETSc.Vec()
x.create(PETSc.COMM_WORLD)
x.setSizes(2)
x.setFromOptions()
x.set(1.0)
r = x.duplicate()

snes = PETSc.SNES()
snes.create(PETSc.COMM_WORLD)
snes.setFunction(formFunction, r)
snes.setFromOptions()
snes.solve(None,x)
x.view()
