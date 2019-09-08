#!/usr/bin/env python

# WARNING:  You will need to edit this file to match your batch system!!

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

intro = '''
Write SLURM batch files for strong scaling study using ch7/minimal.c.  Example:
    ./genstrong.py -email xx@yy.edu -lev 6 -queue t2standard -minP 4 -maxP 64 -pernode 4 -minutes 60
'''

parser = ArgumentParser(description=intro,formatter_class=RawTextHelpFormatter)
parser.add_argument('-email', metavar='EMAIL', type=str,
                    default='USERNAME@alaska.edu', help='email address')
parser.add_argument('-lev', type=int, default=4, metavar='X',
                    help='''refinement level for -snes_grid_sequence; in {4,5,6,7,8}''')
parser.add_argument('-maxP', type=int, default=16, metavar='P',
                    help='''maximum number of MPI processes;
power of 2 like 8,16,64,128,... recommended''')
parser.add_argument('-minP', type=int, default=2, metavar='P',
                    help='''minimum number of MPI processes;
power of 2 like 1,2,4,8,16,... recommended''')
parser.add_argument('-minutes', type=int, default=120, metavar='T',
                    help='''max time in minutes for SLURM job''')
parser.add_argument('-queue', metavar='Q', type=str,
                    default='debug', help='SLURM queue (partition) name')
parser.add_argument('-pernode', type=int, default=2, metavar='K',
                    help='''maximum number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')

args = parser.parse_args()

print('settings: %s queue, %d max tasks per node, %s as email, request time %d minutes'
      % (args.queue,args.pernode,args.email,args.minutes))

m_min = int(np.floor(np.log(float(args.minP)) / np.log(2.0)))
m_max = int(np.floor(np.log(float(args.maxP)) / np.log(2.0)))
Plist = np.round(2.0**np.arange(m_min,m_max+1)).astype(int).tolist()
print('runs (ch7/minimal.c) will use P in'),
print(Plist)

rawpre = r'''#!/bin/bash

#SBATCH --partition=%s
#SBATCH --ntasks=%d
#SBATCH --tasks-per-node=%d
#SBATCH --time=%d
#SBATCH --mail-user=%s
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=%s

# This cluster needs these for some reason.
ulimit -s unlimited
ulimit -l unlimited

# Generate a list of allocated nodes as a machinefile for mpiexec.
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

# Launches the MPI application.
GO="mpiexec -n $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID"

cd $SLURM_SUBMIT_DIR
'''

rawminimal = r'''
# MINIMAL:  solve 2D minimal surface equation
# using grid-sequenced Newton GMRES+GMG solver and 33x33 coarse grid
# with -snes_grid_sequence %d is %dx%d fine grid
# each process has N/P = %d degrees of freedom

$GO ../ch7/minimal -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence %d -snes_fd_color -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view
'''

# pairs  minimaldict[K] = M  where -snes_grid_sequence K generates M x M grid
minimaldict = {  4: 513,
                 5: 1025,
                 6: 2049,
                 7: 4097,
                 8: 8193}

for P in Plist:
    grid = minimaldict[args.lev]
    run = rawminimal % (args.lev,grid,grid,grid*grid/P,args.lev)

    pernode = min(P,args.pernode)
    nodes = P / pernode
    print('  case: run with %d nodes, %d tasks per node, and P=%d processes'
          % (nodes,pernode,P))
    print('        on %d x %d grid; each process has %d degrees of freedom'
          % (grid,grid,grid*grid/P))

    root = 'strong_minimal_%d' % P
    preamble = rawpre % (args.queue,P,pernode,args.minutes,args.email,
                         root + r'.o.%j')

    batchname = root + '.sh'
    print('    writing %s ...' % batchname)
    batch = open(batchname,'w')
    batch.write(preamble)
    batch.write(run)
    batch.close()

