#!/usr/bin/env python

from sys import exit
from pylab import *

# Plot polynomials which come from Richardson iteration with omega=1.

x = linspace(-0.4,2.4,301);

p = []
#p.append(ones(shape(x)))  # omit p_0(x)
p.append(2.0 - x)
p.append(3.0 - 3.0*x + x**2)
p.append(4.0 - 6.0*x + 4.0*x**2 - x**3)
p.append(5.0 -10.0*x +10.0*x**2 - 5.0*x**3 + x**4)

l = []
l.append((0.15,1.48))
l.append((-0.32,3.2))
l.append((-0.25,4.2))
l.append((-0.15,5.3))
l.append((0.2,5.5))

fig = figure(figsize=(7,6))

for n in range(len(p)):
    plot(x,p[n],'k',lw=1.0)
    text(l[n][0],l[n][1],'$q_%d(x)$' % (n+1))
    hold(True)

plot(x[x>0],1.0/x[x>0],'k--',lw=2.0)
text(l[4][0],l[4][1],'$1/x$')

grid(True)
axis([-0.4, 2.4, -0.5, 6.5])
xlabel(r'$x$',fontsize=18.0)

#show()
savefig('richpolyplot.pdf',bbox_inches='tight')
