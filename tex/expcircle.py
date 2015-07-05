#!/usr/bin/env python

from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm

# EXPCIRCLE  Solve intersection of circle and exponential by Newton's method,
# and plot the result.  No linesearch.  The contrast with the PETSc version
# helps to understand linesearch.  This code generates a figure for the
# nonlinear chapter in the book.

b = 2.0
x0 = np.array([1.0, 1.0])

def F(b,x):
    y = x.copy()
    y[0] = (1.0/b) * np.exp(b*x[0]) - x[1]
    y[1] = x[0]**2 + x[1]**2 - 1.0
    return y
 
def J(b,x):
    return np.matrix([[np.exp(b*x[0]), -1.0],
                      [2.0*x[0],       2.0*x[1]]])

# plot curves
fig = plt.figure(figsize=(7,6))
xx = np.linspace(-1.3,1.3,261)
plt.plot(xx, (1.0/b) * np.exp(b*xx), 'k', lw=2.0)
plt.hold(True)
xx = np.linspace(-1.0,1.0,201)
plt.plot(xx, np.sqrt(1.0 - xx*xx), 'k', lw=2.0)

# newton iteration gives x_0, ..., x_3
x = x0
for k in range(3):
    print x
    plt.plot(x[0],x[1],'ko',ms=8.0)
    plt.text(x[0]+0.05,x[1]+0.05,r'$x_%d$' % k,fontsize=20.0)
    x = x - solve(J(b,x),F(b,x))

plt.hold(False)
plt.axis('off')
plt.axis('equal')
plt.axis([-1.3, 1.3, 0.0, 1.5])

#plt.show()
plt.savefig('expcirclebasic.pdf',bbox_inches='tight')

# now plot residual decay
fig = plt.figure(figsize=(7,6))

# newton iteration gives x_0, ..., x_9
x = x0
for k in range(7):
    print norm(F(b,x))
    # PETSc says 1.57e-16, not 0.0 like in python
    plt.semilogy(k,np.maximum(1.57e-16,norm(F(b,x))),'ko',ms=8.0)
    plt.hold(True)
    x = x - solve(J(b,x),F(b,x))
plt.hold(False)
plt.axis([-0.5, 7.5, 1.0e-16, 1.0e1])
plt.grid(True)
plt.xlabel('k',fontsize=20.0)
plt.ylabel('residual norm',fontsize=20.0)
plt.xticks(range(8),fontsize=16.0)
plt.yticks(10.0**np.linspace(-17.0,1.0,7),fontsize=16.0)

#plt.show()
plt.savefig('newtonconvbasic.pdf',bbox_inches='tight')

