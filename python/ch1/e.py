#!/usr/bin/env python3
'''Compute e in parallel with PETSc.'''

import sys
import petsc4py
from petsc4py import PETSc
from mpi4py import MPI
petsc4py.init(sys.argv)

comm = PETSc.COMM_WORLD.tompi4py()
rank = comm.Get_rank()

# compute  1/n!  where n = (rank of process) + 1
localval = 1.0
for i in range(2,rank+1):
    localval = localval / i

# sum the contributions over all processes
globalsum = comm.allreduce(localval, op=MPI.SUM)

# output estimate of e and report on work from each process
PETSc.Sys.Print('e is about %17.15f' % globalsum, comm=PETSc.COMM_WORLD)
if rank > 0:
    q = rank-1
else:
    q = 0
PETSc.Sys.Print('rank %d did %d flops' % (rank,q), comm=PETSc.COMM_SELF)
