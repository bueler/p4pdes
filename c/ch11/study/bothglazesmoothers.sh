#!/bin/bash
set -e

# run with --with-debugging=0 configuration

# generate table comparing GMG smoothers for problem GLAZE using eps=1/200
# and first-order upwinding at all levels

# edited result is put directly in p4pdes-book/chaps/advdif.tex

for SMOOTH in "-mg_levels_pc_type sor" \
              "-mg_levels_pc_type ilu" \
              "-mg_levels_pc_type ilu -mg_levels_pc_factor_levels 1"; do
    for LEV in 5 6 7 8 9 10; do
        CMD="../both -bth_problem glaze -snes_type ksponly -ksp_type bcgs -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -da_refine $LEV $SMOOTH"
        echo $CMD
        $CMD
    done
done

