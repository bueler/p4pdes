#!/bin/sh

#SBATCH --partition=t1standard
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=8
#SBATCH --mail-user=elbueler@alaska.edu
#SBATCH --output=slurm.%j

cd $SLURM_SUBMIT_DIR

# Generate a list of allocated nodes; will serve as a machinefile for mpirun
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

make streams

# Clean up the machinefile
rm ./nodes.$SLURM_JOB_ID

