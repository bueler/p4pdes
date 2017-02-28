#!/bin/bash

#SBATCH --partition=t1standard
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=8
#SBATCH --mail-user=elbueler@alaska.edu
#SBATCH --output=slurm.%j

cd $SLURM_SUBMIT_DIR

# Generate a list of allocated nodes; will serve as a machinefile for mpirun
srun -l /bin/hostname | sort -n | awk '{print $2}' > ./nodes.$SLURM_JOB_ID

LEV=9
echo $LEV
time mpiexec -np $SLURM_NTASKS -machinefile ./nodes.$SLURM_JOB_ID ../ice \
    -da_refine $LEV -pc_type mg -pc_mg_levels $(( $LEV - 2 )) \
    -ice_dtlimits -ice_tf 50000 -ice_dtinit 1.0 \
    -ts_max_snes_failures -1 -ts_adapt_scale_solve_failed 0.9 \
    -ice_dump -snes_converged_reason
done

# Clean up the machinefile
rm ./nodes.$SLURM_JOB_ID

