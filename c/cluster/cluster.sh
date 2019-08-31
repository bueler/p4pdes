#!/bin/bash

# WARNING:  You will need to edit this file to match your batch system!!

# This is a sample batch file for running jobs on a linux cluster.

# The goals for the runs below, and their various sizes, is that
# that the total run time is less than 15 minutes for P >= 4 processes.
# With a bigger machine (e.g. >100 processes) consider further refinement of
# grids.  The total memory usage is less than 16 Gb.

# In parent directory ../c/, set PETSC_ARCH to a --with-debugging=0 PETSc
# configuration.  Then run "make distclean" to clean out old executables.
# Then run "make test" to make sure executables in ../ch*/ get built.

# For each run a -log_view is generated in a file cluster.xxx.  This can be
# analyzed for performance data.  For example do
#   grep "Flop:  " p4pdes.xxx
#   grep "Time (sec):" p4pdes.xxx
# to see total flops and max times of runs.

# Many details below are particular to the Slurm batch system:
#     https://slurm.schedmd.com/
# Normally we use a better partition and more tasks.

#SBATCH --partition=debug
#SBATCH --ntasks=12
#SBATCH --tasks-per-node=4
#SBATCH --mail-user=<USERNAME>@alaska.edu  # CHANGE TO REAL EMAIL FOR RUN
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=p4pdes.o.%j

# This cluster needs these for some reason.
ulimit -s unlimited
ulimit -l unlimited

# Get streams info for these processes and then head back to submit directory.
# (Submit from c/cluster/ for this to work.)
cd $PETSC_DIR
make streams NPMAX=$SLURM_NTASKS
cd $SLURM_SUBMIT_DIR

# Generate a list of allocated nodes; will serve as a machinefile for mpirun.
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

# Launch the MPI application.  Tested with mpich2.
GO="mpiexec -n $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID"

#-------------------  PARALLEL RUNS -------------------

# PATTERN:  solve 2D, time-dependent, coupled diffusion-reaction equations
# using ARKIMEX time-stepping and GMRES+BJACOBI+ILU to solve implicit steps
# -da_refine 6 is 256x256; does ~100 time steps
$GO ../ch5/pattern -da_refine 6 -ptn_phi 0.05 -ptn_kappa 0.063 -ts_max_time 2000 -ptn_noisy_init 0.15 -ts_monitor -log_view

# FISH:  solve 3D Poisson equation
# using optimal CG+GMG solver
# -da_refine 7 is 257x257x257
# ~ 12 Gb memory
# coarse grid is 9x9x9 so should work up to several hundred processors
$GO ../ch6/fish -fsh_dim 3 -da_refine 7 -pc_mg_levels 6 -pc_type mg -snes_type ksponly -ksp_converged_reason -log_view

# MINIMAL:  solve 2D minimal surface equation
# using optimal grid-sequenced Newton GMRES+GMG solver
# 33x33 base grid with -snes_grid_sequence 6 is 2049x2049 finest grid
# ~ 8 Gb memory
# coarse grid is 33x33 so should work up to several hundred processors
$GO ../ch7/minimal -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence 6 -snes_fd_color -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view

# ADVECT:  solve 2D, time-dependent advection equation
# using RK3bs (explicit) solver
# -da_refine 8 is 1280x1280; does ~1200 time steps
$GO ../ch11/advect -da_refine 8 -ts_max_time 0.5 -ts_rk_type 3bs -ts_monitor -log_view

# OBSTACLE:  solve 2D inequality-constrained free-boundary problem
# using optimal grid-sequenced bound-constrained Newton CG+GMG solver
# 33x33 base grid with -snes_grid_sequence 6 is 2049x2049 finest grid
# ~ 8 Gb memory
# coarse grid is 33x33 so should work up to several hundred processors
$GO ../ch12/obstacle -da_grid_x 33 -da_grid_y 33 -snes_grid_sequence 6 -snes_converged_reason -snes_monitor -ksp_converged_reason -pc_type mg -log_view

#------------------------------------------------------

# Clean up the machinefile.
rm ./nodes.$SLURM_JOB_ID

