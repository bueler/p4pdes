#!/usr/bin/env python3
'''Newton's method for a two-variable system.  Implements analytical
Jacobian and a struct to hold a parameter.'''

import sys
from numpy import exp
import petsc4py
petsc4py.init(sys.argv)  # must come before "import PETSc"
from petsc4py import PETSc

class AppCtx:
    '''Holds parameter b.  Supplies call-back functions.'''

    def __init__(self, b=1.0):
        self.b = b

    def formFunction(self, _snes, X, F):
        '''Evaluate F(X).'''
        ax = X.getArray(readonly=1)
        af = F.getArray(readonly=0)
        af[0] = (1.0 / self.b) * exp(self.b * ax[0]) - ax[1]
        af[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0

    def formJacobian(self, _snes, X, Jmat, Pmat):
        '''Evaluate J(X) where J = F'.'''
        ax = X.getArray(readonly=1)
        row = [0,1]
        col = row
        v = [exp(self.b * ax[0]),  -1.0,
             2.0 * ax[0],          2.0 * ax[1]]
        Pmat.setValues(row,col,v)
        Pmat.assemble()
        if Jmat != Pmat:
            Jmat.assemble()

ctx = AppCtx(b=2.0)

x = PETSc.Vec()
x.create(PETSc.COMM_WORLD)
x.setSizes(2)
x.setFromOptions()
x.set(1.0)
r = x.duplicate()

J = PETSc.Mat()
J.create(PETSc.COMM_WORLD)
J.setSizes((2,2))
J.setFromOptions()
J.setUp()

snes = PETSc.SNES()
snes.create(PETSc.COMM_WORLD)
snes.setFunction(ctx.formFunction, r)
snes.setJacobian(ctx.formJacobian, J)
snes.setFromOptions()
snes.solve(None,x)
x.view()
