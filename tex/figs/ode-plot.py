#!/usr/bin/env python

# see c/ch6/plottrajectory.py for context/help

import numpy as np
import matplotlib.pyplot as plt

outname = 'ode.pdf'

t = np.arange(0.0,20.0,0.05)
y0 = t - np.sin(t)
y1 = 1.0 - np.cos(t)

plt.plot(t,y0,'k',lw=3.0)
plt.plot(t,y1,'k--',lw=4.0)
plt.xlabel('t',fontsize=20.0)

print "writing %s ..." % outname
plt.savefig(outname,bbox_inches='tight')

