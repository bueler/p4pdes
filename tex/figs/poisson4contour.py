#!/usr/bin/env python

# shows steps to generate poisson4contour.pdf
# do:
#     ./poisson -da_refine 4 -ksp_monitor_solution binary:foo.dat
# which shows the solution at 0 .. 174 Krylov iterations

import PetscBinaryIO
import numpy as np
import matplotlib.pyplot as plt

io = PetscBinaryIO.PetscBinaryIO()
objects = io.readBinaryFile('foo.dat')
x = np.linspace(0,1,129)
plt.contour(x,x,-objects[174].reshape((129,129)),15,colors='k')
plt.axis('tight')
plt.axis('equal')  # actually made it square by hand
plt.savefig('foo.pdf')

