#!/usr/bin/env python

# example:
#   $ ./genweak.py -email elbueler@alaska.edu -queue debug -maxP 16 -pernode 6 -time 30

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

parser = ArgumentParser(description='''
Write SLURM batch files for weak scaling study using ch7/minimal.c in 2D and
ch6/fish.c in 3D.''',
    formatter_class=RawTextHelpFormatter)
parser.add_argument('-email', metavar='EMAIL', type=str,
                    default='USERNAME@alaska.edu', help='email address')
parser.add_argument('-maxP', type=int, default=4, metavar='P',
                    help='''maximum number of MPI processes;
power of 4 like 16,64,256,1024,4096,... recommended''')
parser.add_argument('-queue', metavar='QUEUE', type=str,
                    default='debug', help='SLURM queue (partition) name')
parser.add_argument('-pernode', type=int, default=2, metavar='K',
                    help='''number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')
parser.add_argument('-time', type=int, default=120, metavar='T',
                    help='''max time in minutes for SLURM job''')
args = parser.parse_args()

print('settings for SLURM: %s queue, %d tasks per node, %s as email'
      % (args.queue,args.pernode,args.email))

m2D = int(np.floor(np.log(float(args.maxP)) / np.log(4.0)))
Plist2D = np.round(4.0**np.arange(m2D+1)).astype(int).tolist()
print('2D runs (ch7/minimal.c) will use P in'),
print(Plist2D)

m3D = int(np.floor(np.log(float(args.maxP)) / np.log(8.0)))
Plist3D = np.round(8.0**np.arange(m3D+1)).astype(int).tolist()
print('3D runs (ch6/fish.c) will use P in'),
print(Plist3D)

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

# Get streams info for these processes.
cd $PETSC_DIR
make streams NPMAX=$SLURM_NTASKS
cd $SLURM_SUBMIT_DIR

# Generate a list of allocated nodes as a machinefile for mpiexec.
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

# Launches the MPI application.
GO="mpiexec -n $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID"
'''

rawfish = r'''
# FISH:  solve 3D Poisson equation
# using optimal CG+GMG solver and 9x9x9 coarse grid
# with -da_refine %d is %dx%dx%d fine grid
# each process has N/P = %d degrees of freedom

$GO ../ch6/fish -fsh_dim 3 -da_grid_x 9 -da_grid_y 9 -da_grid_z 9 -da_refine %d -pc_type mg -snes_type ksponly -ksp_converged_reason -log_view
'''
fishdict = {  1: (3,65),
              8: (4,129),
             64: (7,257),
            512: (8,513),
           4096: (9,1025)}

rawminimal = r'''
# MINIMAL:  solve 2D minimal surface equation
# using grid-sequenced Newton GMRES+GMG solver and 33x33 coarse grid
# with -snes_grid_sequence %d is %dx%d fine grid
# each process has N/P = %d degrees of freedom

$GO ../ch7/minimal -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence %d -snes_fd_color -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view
'''
minimaldict = {  1: (4,513),
                 4: (5,1025),
                16: (6,2049),
                64: (7,4097),
               256: (8,8193),
              1024: (9,16385),
              4096: (10,65537)}

for dim in [2, 3]:
    if dim == 2:
        Plist = Plist2D
        code = 'minimal'
        xdict = minimaldict
    else:
        Plist = Plist3D
        code = 'fish'
        xdict = fishdict
    for P in Plist:
        nodes = P / args.pernode
        rlev = xdict[P][0]  # refinement level
        grid = xdict[P][1]
        if dim == 2:
            wrun = rawminimal % (rlev,grid,grid,grid*grid/P,rlev)
        else:
            wrun = rawfish % (rlev,grid,grid,grid,grid*grid*grid/P,rlev)

        print('  case: run %s with %d nodes and P=%d processes on %d^%d grid'
              % (code,nodes,P,grid,dim))

        root = 'weak_%s_%d' % (code,P)
        preamble = rawpre % (args.queue,P,args.pernode,args.time,args.email,
                             root + r'.o.%j')

        batchname = root + '.sh'
        print('    writing %s ...' % batchname)
        batch = open(batchname,'w')
        batch.write(preamble)
        batch.write(wrun)
        batch.close()

