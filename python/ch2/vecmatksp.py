#!/usr/bin/env python3
'''Solve a 4x4 linear system using KSP.'''

import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
from petsc4py import PETSc

j = [0,1,2,3]
ab = [7.0,1.0,1.0,3.0]
aA = [[1.0,2.0,3.0,0.0],
      [2.0,1.0,-2.0,-3.0],
      [-1.0,1.0,1.0,0.0],
      [0.0,1.0,1.0,-1.0]]

b = PETSc.Vec()
b.create(PETSc.COMM_WORLD)
b.setSizes(4)
b.setFromOptions()
b.setValues(j,ab)
b.assemblyBegin()
b.assemblyEnd()

A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes((4,4))
A.setFromOptions()
A.setUp()
for i in range(4):
    A.setValues([i,],j,aA[i])
A.assemblyBegin()
A.assemblyEnd()

ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(A=A,P=A)
ksp.setFromOptions()
x = b.duplicate()
ksp.solve(b,x)
x.view()

