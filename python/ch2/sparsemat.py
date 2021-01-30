#!/usr/bin/env python3
'''Assemble a Mat sparsely.'''

import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
from petsc4py import PETSc

i1 = [0, 1, 2]
j1 = [0, 1, 2]
i2 = 3
j2 = [1, 2, 3]
i3 = 1
j3 = 3
aA1 = [1.0,  2.0,  3.0,
       2.0,  1.0, -2.0,
       -1.0,  1.0,  1.0]
aA2 = [1.0,  1.0, -1.0]
aA3 = -3.0

A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes((4,4))
A.setFromOptions()
A.setUp()
A.setValues(i1,j1,aA1)
A.setValues(i2,j2,aA2)
A.setValue(i3,j3,aA3)
A.assemblyBegin()
A.assemblyEnd()

