#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

names = ['ErrorsEvalsTimes','snes-fd-ErrorsEvalsTimes']
for n in range(2):
    eet = np.loadtxt('../timing/unfem/' + names[n])
    N = len(eet)
    eet = np.reshape(eet,(N/4,4))
    h = eet[:,0]
    err = eet[:,1]
    evals = np.array(eet[:,2],dtype=int)
    time = eet[:,3]
    plt.loglog(h,err,'ko',markersize=16.0,markerfacecolor='w')
    plt.grid(True)
    plt.xlabel(r'$h$',fontsize=20.0)
    plt.ylabel('relative error',fontsize=18.0)
    plt.show()

#filename='plap-conv.pdf'
#fig = plt.figure(figsize=(7,6))
#plt.hold(True)
#plt.text(0.012,4.0e-6,r'$O(h^{%.4f})$' % p[0],fontsize=20.0)
#plt.axis((1.0e-3,0.35,1.0e-7,3.0e-2))
#plt.xticks(np.array([0.001, 0.01, 0.1]),
#                    (r'$0.001$',r'$0.01$',r'$0.1$'),
#                    fontsize=16.0)
#plt.yticks(np.array([1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]),
#                    (r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'),
#                    fontsize=16.0)
##plt.show()
#print "writing %s ..." % filename
#plt.savefig(filename,bbox_inches='tight')

