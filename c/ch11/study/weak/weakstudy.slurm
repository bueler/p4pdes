#!/bin/bash

#SBATCH --partition=t1standard
#SBATCH --ntasks=64
#SBATCH --tasks-per-node=8
#SBATCH --mail-user=elbueler@alaska.edu
#SBATCH --output=slurm.%j

cd $SLURM_SUBMIT_DIR

LEV=9
NP=1              # NP_k = 4 * NP_{k-1}
for KK in 0 1 2 3; do
    echo $NP
    echo $LEV
    time mpiexec -np $NP ../../ice \
        -ice_verif 2 -ts_type beuler -ice_tf 10.0 -ice_dtinit 10.0 -da_refine $LEV \
        -snes_monitor -snes_converged_reason -ksp_converged_reason \
        -pc_type mg -pc_mg_levels $(( $LEV - 1 ))
    LEV=$(( $LEV + 1 ))
    NP=$(( $NP * 4 ))
done

