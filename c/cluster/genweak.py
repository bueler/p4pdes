#!/usr/bin/env python

# this version uses ch6/fish and ch7/minimal

from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np

parser = ArgumentParser(description='Write SLURM batch files for weak scaling study.',
    formatter_class=RawTextHelpFormatter)
parser.add_argument('-email', metavar='EMAIL', type=str,
                    default='USERNAME@alaska.edu', help='email address')
parser.add_argument('-maxP', type=int, default=4, metavar='P',
                    help='''maximum number of MPI processes;
power of 4 like 16,64,256,1024,4096,... recommended''')
parser.add_argument('-queue', metavar='QUEUE', type=str,
                    default='debug', help='queue name')
parser.add_argument('-pernode', type=int, default=2, metavar='K',
                    help='''number of MPI processes to assign to each node;
small value may increase streams bandwidth and performance''')
#parser.add_argument('-quad', action='store_true', default=False,
#                    help='use quadrilateral finite elements')
args = parser.parse_args()
#print(args)

print('using %s queue and %s ...' % (args.queue,args.email))

m = int(round(np.log(float(args.maxP)) / np.log(4.0)))
Plist = np.round(4.0**np.arange(m+1)).astype(int).tolist()
print('using P in'),
print(Plist),
print('...')

for P in Plist:
    nodes = P / args.pernode
    print('  case: %d nodes and %d tasks per node for P=%d processes:'
          % (nodes,args.pernode,P))

    preamble = r'''#!/bin/bash

#SBATCH --partition=%s
#SBATCH --ntasks=%d
#SBATCH --tasks-per-node=%d
#SBATCH --mail-user=%s
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

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

''' % (args.queue,P,args.pernode,args.email)

    minimaldict = { 1: (4,513),
                    4: (5,1025),
                   16: (6,2049),
                   64: (7,4096)}
    mseq = minimaldict[P][0]
    mgrid = minimaldict[P][1]
    mrun = r'''
# MINIMAL:  solve 2D minimal surface equation
# using grid-sequenced Newton GMRES+GMG solver and 33x33 coarse grid
# with -snes_grid_sequence %d is %dx%d finest grid

$GO ../ch7/minimal -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence %d -snes_fd_color -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view
''' % (mseq,mgrid,mgrid,mseq)

    batchname = 'weak%d.sh' % P
    print('    writing %s' % batchname)
    batch = open(batchname,'w')
    batch.write(preamble)
    batch.write(mrun)
    batch.close()

