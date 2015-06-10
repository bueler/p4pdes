#!/usr/bin/env python

from sys import exit
from pylab import *

# Plot polynomials which come from Richardson iteration with omega=1.

x = linspace(-0.5,2.5,301);

p = []
#p.append(ones(shape(x)))  # omit p_0(x)
p.append(2.0 - x)
p.append(3.0 - 3.0*x + x**2)
p.append(4.0 - 6.0*x + 4.0*x**2 - x**3)
p.append(5.0 -10.0*x +10.0*x**2 - 5.0*x**3 + x**4)

fig = figure(figsize=(7,6))

for n in range(len(p)):
    plot(x,p[n],'k',lw=1.0)
    hold(True)

plot(x[x>0],1.0/x[x>0],'k--',lw=2.0)

grid(True)
axis([-0.5, 2.5, -0.5, 6.5])
xlabel(r'$x$',fontsize=18.0)

#show()
savefig('richpolyplot.pdf',bbox_inches='tight')
