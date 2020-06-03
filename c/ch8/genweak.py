#!/usr/bin/env python3

# WARNING:  You will need to edit this file to match your batch system!!

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

intro = '''
Write SLURM batch files for weak scaling study using ch7/minimal.c.  Examples:
    ./genweak.py -email elbueler@alaska.edu -queue debug -maxP 1 -minutes 60
    ./genweak.py -email elbueler@alaska.edu -queue debug -minP 4 -maxP 4 -pernode 2 -minutes 60
    ./genweak.py -email elbueler@alaska.edu -queue debug -minP 4 -maxP 4 -pernode 4 -minutes 60
    ./genweak.py -email elbueler@alaska.edu -queue t2standard -minP 4 -maxP 4 -pernode 1 -minutes 60
    ./genweak.py -email elbueler@alaska.edu -queue t2standard -minP 16 -maxP 64 -pernode 4 -minutes 60
    ./genweak.py -email elbueler@alaska.edu -queue t2standard -minP 16 -maxP 256 -pernode 8 -minutes 60
Solves 2D minimal surface equation using grid-sequenced Newton GMRES+GMG solver
and 33x33 coarse grid.  Each process gets a 1024x1024 grid with N/P = 1.05e6.
'''

parser = ArgumentParser(description=intro, formatter_class=RawTextHelpFormatter)
parser.add_argument('-email', metavar='EMAIL', type=str,
                    default='USERNAME@alaska.edu', help='email address')
parser.add_argument('-maxP', type=int, default=4, metavar='P',
                    help='''maximum number of MPI processes;
power of 4 like 4,16,64,256,1024,... recommended''')
parser.add_argument('-minP', type=int, default=1, metavar='P',
                    help='''minimum number of MPI processes;
power of 4 like 1,4,16,64,256,... recommended''')
parser.add_argument('-minutes', type=int, default=60, metavar='T',
                    help='''max time in minutes for SLURM job''')
parser.add_argument('-queue', metavar='Q', type=str,
                    default='debug', help='SLURM queue (partition) name')
parser.add_argument('-pernode', type=int, default=2, metavar='K',
                    help='''maximum number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')
parser.add_argument('-streams', action='store_true', default=False,
                    help='include "make streams" before run (but may hang on attempt to use python?)')

args = parser.parse_args()

print('settings: %s queue, %d max tasks per node, %s as email, request time %d minutes'
      % (args.queue,args.pernode,args.email,args.minutes))

m_min = int(np.floor(np.log(float(args.minP)) / np.log(4.0)))
m_max = int(np.floor(np.log(float(args.maxP)) / np.log(4.0)))
Plist = np.round(4.0**np.arange(m_min,m_max+1)).astype(int).tolist()
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

rawstreams= r'''
# Get streams info for these processes.
cd $PETSC_DIR
make streams NPMAX=$SLURM_NTASKS
cd $SLURM_SUBMIT_DIR
'''

rawminimal = r'''
# MINIMAL:  solve 2D minimal surface equation
# using grid-sequenced Newton GMRES+GMG solver and 33x33 coarse grid
# with -snes_grid_sequence %d is %dx%d fine grid
# each process has N/P = %d degrees of freedom

$GO ../ch7/minimal -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence %d -snes_fd_color -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view
'''
minimaldict = {  1: (5,1025),
                 4: (6,2049),
                16: (7,4097),
                64: (8,8193),
               256: (9,16385)}

for P in Plist:
    rlev = minimaldict[P][0]  # refinement level
    grid = minimaldict[P][1]
    wrun = rawminimal % (rlev,grid,grid,grid*grid/P,rlev)

    pernode = min(P,args.pernode)
    nodes = P / pernode
    print('  case: %d nodes, %d tasks per node, and P=%d processes on %dx%d grid'
          % (nodes,pernode,P,grid,grid))

    root = 'weak_minimal_%s_%d_%d' % (args.queue[:2],P,pernode)
    preamble = rawpre % (args.queue,P,pernode,args.minutes,args.email,
                         root + r'.o.%j')

    batchname = root + '.sh'
    print('    writing %s ...' % batchname)
    batch = open(batchname,'w')
    batch.write(preamble)
    if args.streams:
        batch.write(rawstreams)
    batch.write(wrun)
    batch.close()

