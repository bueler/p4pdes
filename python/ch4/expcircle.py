#!/usr/bin/env python3
'''Newton's method for a two-variable system.  No analytical Jacobian.
Run with -snes_fd (default) or -snes_mf.'''

import sys
from numpy import exp
import petsc4py
petsc4py.init(sys.argv)  # must come before "import PETSc"
from petsc4py import PETSc

def formFunction(snes, X, F):
    b = 2.0
    x = X.getArray(readonly=1)
    f = F.getArray(readonly=0)
    f[0] = (1.0 / b) * exp(b * x[0]) - x[1]
    f[1] = x[0] * x[0] + x[1] * x[1] - 1.0

x = PETSc.Vec()
x.create(PETSc.COMM_WORLD)
x.setSizes(2)
x.setFromOptions()
x.set(1.0)
r = x.duplicate()

snes = PETSc.SNES()
snes.create(PETSc.COMM_WORLD)
snes.setFunction(formFunction,r)
snes.setFromOptions()
snes.solve(None,x)
x.view()
