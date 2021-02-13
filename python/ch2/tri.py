#!/usr/bin/env python3
'''Solve a tridiagonal system of arbitrary size.  Option prefix = tri_.'''

import sys
import numpy as np
import petsc4py
petsc4py.init(sys.argv)  # must come before "import PETSc"
from petsc4py import PETSc

Opt = PETSc.Options(prefix='tri_')
m = Opt.getInt('m',default=4)

x = PETSc.Vec()
x.create(PETSc.COMM_WORLD)
x.setSizes(m)
x.setFromOptions()
b = x.duplicate()
xexact = x.duplicate()

A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes((m,m))
A.setOptionsPrefix("a_")
A.setFromOptions()
A.setUp()

Istart,Iend = A.getOwnershipRange()
for i in range(Istart,Iend):
    if i == 0:
        A.setValues([0,],[0,1],[3.0,-1.0])
    else:
        v = [-1.0,3.0,-1.0]
        j = [i-1,i,i+1]
        if i == m-1:
            A.setValues([m-1,],j[0:2],v[0:2])
        else:
            A.setValues([i,],j,v)
    xval = np.exp(np.cos(float(i)))
    xexact.setValues([i,],[xval,])
A.assemblyBegin()
A.assemblyEnd()
xexact.assemblyBegin()
xexact.assemblyEnd()
A.mult(xexact,b)

ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(A=A,P=A)
ksp.setFromOptions()
ksp.solve(b,x)

x.axpy(-1.0,xexact)
errnorm = x.norm(norm_type=PETSc.NormType.NORM_2)
PETSc.Sys.Print('error for m = %d system is |x-xexact|_2 = %.1e' \
                % (m,errnorm))
