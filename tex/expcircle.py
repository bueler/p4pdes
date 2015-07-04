#!/usr/bin/env python

from sys import exit
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve

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
xx = np.linspace(-1.5,1.5,301)
plt.plot(xx, (1.0/b) * np.exp(b*xx), 'k', lw=2.0)
plt.hold(True)
xx = np.linspace(-1.0,1.0,201)
plt.plot(xx, np.sqrt(1.0 - xx*xx), 'k', lw=2.0)

# newton iteration gives x_0, ..., x_3
x = x0
for k in range(3):
    print x
    plt.plot(x[0],x[1],'ko',ms=8.0)
    plt.text(x[0]+0.05,x[1]+0.05,r'$x_%d$' % k,fontsize=18.0)
    x = x - solve(J(b,x),F(b,x))

plt.hold(True)
plt.axis('off')
plt.axis('equal')
plt.axis([-1.5, 1.5, 0.0, 1.5])


#plt.show()
plt.savefig('expcirclebasic.pdf',bbox_inches='tight')

