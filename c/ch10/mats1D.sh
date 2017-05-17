#!/bin/bash
set -e
set +x

ln -sf ~/petsc/bin/PetscBinaryIO.py
ln -sf ~/petsc/bin/petsc_conf.py
./well -snes_converged_reason -da_refine 3 -mat_view binary:matstag.dat
./well -snes_converged_reason -da_refine 3 -well_scheme regular -snes_fd_color -mat_view binary:matregu.dat

