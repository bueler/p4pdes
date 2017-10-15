#!/usr/bin/env python

# run as:
#    ln -s ../sneskspplot.py   # FIXME
#    ./fish -da_refine 4 -snes_type ksponly -ksp_type richardson -pc_type sor -pc_sor_forward -ksp_monitor -ksp_max_it 30 -fsh_initial_type random &> stagnate.txt
#    ./stagnatefig.py

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

sys.path.append('../')   # FIXME
import sneskspplot as skp

snesdata, kspdata = skp.readskfile(open('stagnate.txt', 'r'))
skp.genskfigure(snesdata, kspdata, showsnes=False, showksp=True, ksptrue=False)

plt.xlabel('KSP iteration',fontsize=18.0)
plt.ylabel('residual norm',fontsize=18.0)
plt.xticks(np.linspace(0.0,30.0,11))
plt.yticks([0.5, 1.0, 5.0, 10.0])
ax = plt.gca()
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.grid(False)
plt.savefig('stagnate.png')
