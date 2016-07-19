#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def feuler(z):
    return 1.0 + z

def frk2a(z):
    return 1.0 + z + z*z/2.0

def fbeuler(z):
    return 1.0 / (1.0 - z)

def ftrap(z):
    return (1.0 + z/2.0) / (1.0 - z/2.0)

x, y = np.linspace(-3.0,2.5,250), np.linspace(-2.2,2.2,200)
xx, yy = np.meshgrid(x,y)
zz = xx + yy * 1j

fc = [(fbeuler,'0.8'),
      (ftrap,'0.6'),
      (frk2a,'0.4'),
      (feuler,'0.2')]
fig = plt.figure(figsize=(7,6))
for F,c in fc:
    plt.contour(x,y,abs(F(zz)),[1],colors='k',linewidths=2.0)
    plt.hold(True)
    if len(c) > 0:
        plt.contourf(x,y,abs(F(zz)),levels=[0.0,1.0],colors=c,extend='min')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.grid(True)
plt.hold(False)
plt.axis([-2.6,2.4,-2.2,2.2])
plt.savefig('absstabregions.pdf',bbox_inches='tight')

fig = plt.figure(figsize=(7,6))
tf = 7.0
lam = -1.0
t = np.linspace(0.0,tf,101.0)
plt.plot(t,np.exp(lam*t),'k',label='exact')
plt.hold(True)
hctfwl = [(0.5,'ko--',1.0,feuler,'$h=0.5$ Euler'),
          (2.2,'ko--',2.5,feuler,'$h=2.2$ Euler'),
          (2.2,'ko:', 4.0,frk2a, '$h=2.2$ RK2a')]
for h,ltype,lwidth,f,label in hctfwl:
    N = int(tf / h) + 1
    z = h * lam
    t = np.linspace(0.0,h*N,N+1)
    y = np.ones(N+1)
    for l in range(N):
        y[l+1] = f(z) * y[l]
    plt.plot(t,y,ltype,lw=lwidth,label=label)
plt.grid(True)
plt.hold(False)
plt.axis([0.0,tf,-3.0,2.5])
plt.xlabel('t')
plt.ylabel('y')
plt.legend(loc='lower left')
plt.savefig('absstabfail.pdf',bbox_inches='tight')

#plt.show()

