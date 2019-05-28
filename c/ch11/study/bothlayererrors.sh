#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# verification figure for classical GS-smoothed V-cycle
# multigrid on 2D advection-diffusion in advection-dominated eps=1/200 case
# first-order upwinding at all levels

# edited result is p4pdes-book/figs/bothlayererrors.txt which is
# loaded by figure script p4pdes-book/figs/bothlayererrors.py

for LEV in 5 6 7 8 9 10 11; do
    ../both -bth_eps 0.005 -bth_limiter centered -bth_none_on_peclet -bth_problem layer \
         -snes_type ksponly -ksp_converged_reason -pc_type mg \
         -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward \
         -ksp_rtol 1.0e-10 -da_refine $LEV
done

