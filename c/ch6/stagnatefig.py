#!/usr/bin/env python

# FIXME still needs work: grid, yticks, xticks

# run as:
#    ln -s ../sneskspplot.py
#    ./fish -da_refine 4 -snes_type ksponly -ksp_type richardson -pc_type sor -pc_sor_forward -ksp_monitor -ksp_max_it 30 -fsh_initial_type random &> stagnate.txt
#    ./stagnatefig.py --nosnes --noshow stagnate.txt

import numpy as np
import matplotlib.pyplot as plt

import sneskspplot

plt.xlabel('KSP iteration',fontsize=18.0)
plt.grid(False)
plt.grid(axis='y')
plt.savefig('stagnate.png')
