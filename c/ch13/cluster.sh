#!/bin/bash

# This is a sample batch file for running jobs on a linux cluster.  A goal is that the
# total run time is less than an hour for > 8 processes.  If you have a bigger machine
# (e.g. >100 processes) consider refining the grids more in the runs below.

# In parent directory ../c/, set PETSC_ARCH to a --with-debugging=0 PETSc configuration.
# Then run "make distclean" to clean out old executables.  Then run "make test" to make sure
# executables in ../ch*/ get built.

# For each run a -log_view is generated.  These can be analyzed for performance
# data.  For example do
#   grep "(sec):" cluster.xxxx
# to see total times of runs.

# You will need to edit this file to match your batch system, and for each run!!
# Many details below are particular to the Slurm batch system (see https://slurm.schedmd.com/).

#SBATCH --partition=debug
#SBATCH --ntasks=8
#SBATCH --tasks-per-node=4
#SBATCH --mail-user=elbueler@alaska.edu   # FIXME don't release this way; I'll get spammed by slurm!
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=cluster.o.%j

# Submit from c/ch13/ for this to work.
cd $SLURM_SUBMIT_DIR

# This cluster needs these for some reason.
ulimit -s unlimited
ulimit -l unlimited

# Generate a list of allocated nodes; will serve as a machinefile for mpirun.
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

# Launch the MPI application.  Tested with mpich2.
GO="mpiexec -n $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID"

#---------------------------------------------------------------------------------
# -da_refine 6 is 256x256; this does ~230 time steps
$GO ../ch5/pattern -da_refine 6 -ptn_phi 0.05 -ptn_kappa 0.063 -ts_final_time 5000 -ptn_noisy_init 0.15 -ts_monitor -log_view

# -da_refine 8 is 513x513x513; uses ~6Gb memory
$GO ../ch6/fish -fsh_dim 3 -da_refine 8 -pc_type mg -snes_type ksponly -ksp_converged_reason -log_view

# -da_refine 8 is 1280x1280; this does ~2400 time steps
$GO ../ch9/advect -da_refine 8 -ts_final_time 1.0 -ts_rk_type 3bs -ts_monitor -log_view

# 33x33 base grid with -snes_grid_sequence 6 is 2049x2049 finest grid
$GO ../ch12/obstacle -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence 6 -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view

#FIXME add an unstructured from Chapter 10?
#---------------------------------------------------------------------------------

# Clean up the machinefile.
rm ./nodes.$SLURM_JOB_ID

