#!/usr/bin/env python3
'''Load a matrix  A  and right-hand-side  b  from binary files (PETSc format).
Then solve the system  A x = b  using KSPSolve().
Example.  First save a system from tri.py:
  ./tri.py -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
then load it and solve:
  ./loadsolve -fA A.dat -fb b.dat
To time the solution read the third printed number:
  ./loadsolve -fA A.dat -fb b.dat -log_view |grep KSPSolve
(This is a simpler code than src/ksp/ksp/examples/tutorials/ex10.c.)'''

# small system example w/o RHS:
# ./tri.py -ksp_view_mat binary:A.dat
# ./loadsolve.py -fA A.dat -ksp_view_mat -ksp_view_rhs

# large tridiagonal system (m=10^7) example:
# ./tri.py -tri_m 10000000 -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
# ./loadsolve.py -fA A.dat -fb b.dat -log_view |grep KSPSolve

import sys, petsc4py
petsc4py.init(sys.argv)
import numpy as np
from petsc4py import PETSc

Opt = PETSc.Options()
nameA = Opt.getString('-fA', default='')
nameb = Opt.getString('-fb', default='')
verbose = Opt.getBool('-verbose', default=False)

if verbose:
    PETSc.Sys.Print('reading matrix from %s ...' % nameA)
fileA = PETSc.Viewer().createBinary(nameA, 'r')
A = PETSc.Mat().load(fileA)
m,n = A.getSize()
if verbose:
    PETSc.Sys.Print('matrix has size m x n = %d x %d ...' % (m,n))
if m != n:
    raise ValueError('only works for square matrices')

if len(nameb) == 0:
    if verbose:
        PETSc.Sys.Print('right-hand-side vector b not provided ... using zero vector of length %d' % m)
    b = PETSc.Vec()
    b.setSizes(m)
    b.zeroEntries()  
else:
    if verbose:
        PETSc.Sys.Print('reading vector from %s ...' % nameb)
    fileb = PETSc.Viewer().createBinary(nameb, 'r')
    b = PETSc.Vec().load(fileb)
    mb = b.getSize()
    if mb != m:
        raise ValueError('size of matrix and vector do not match')

ksp = PETSc.KSP()
ksp.create(PETSc.COMM_WORLD)
ksp.setOperators(A=A,P=A)
ksp.setFromOptions()

x = b.duplicate()
x.zeroEntries()
ksp.solve(b,x)
#x.view()

