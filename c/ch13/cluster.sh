#!/bin/bash

# This is a sample batch file for running jobs on a linux cluster.

# In parent directory ../c/ run "make test" first so executables ../ch*/CODE
# get built.  This script runs these executables:
#   * ../ch5/pattern
#   * ../ch6/fish
#   * ../ch9/advect
#   * ../ch12/obstacle
# For each of these a -log_view is generated, and analyzed for performance data.

# You will need to edit this file to match your batch system.


#SBATCH --partition=debug
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=4
#SBATCH --mail-user=elbueler@alaska.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=study.%j

ulimit -s unlimited
ulimit -l unlimited

cd $SLURM_SUBMIT_DIR

# Generate a list of allocated nodes; will serve as a machinefile for mpirun
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

# Launch the MPI application
#mpirun -np $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID ./<APPLICATION>

#../ch5/pattern
../ch6/fish -fsh_dim 3 -da_refine 4
#../ch9/advect
#../ch12/obstacle

# Clean up the machinefile
rm ./nodes.$SLURM_JOB_ID

