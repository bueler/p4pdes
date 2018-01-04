#!/bin/bash
set -e
set +x

ln -sf ~/petsc/bin/PetscBinaryIO.py
ln -sf ~/petsc/bin/petsc_conf.py
./well -snes_converged_reason -da_refine 2 -mat_view binary:matstag2.dat
./well -snes_converged_reason -da_refine 2 -well_scheme regular -snes_fd_color -mat_view binary:matregu2.dat
./well -snes_converged_reason -da_refine 3 -mat_view binary:matstag3.dat
./well -snes_converged_reason -da_refine 3 -well_scheme regular -snes_fd_color -mat_view binary:matregu3.dat
./well -snes_converged_reason -da_refine 4 -mat_view binary:matstag4.dat
./well -snes_converged_reason -da_refine 4 -well_scheme regular -snes_fd_color -mat_view binary:matregu4.dat

