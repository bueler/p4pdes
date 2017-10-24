#!/bin/bash
set -e

# for anisotropic form of Poisson in 2D, with random initial, comparing
# Chebyshev, SSOR, and GS smoothers, measure reduction in norm from KSPITS
# V-cycles

#   * timings do not matter (but using --with-debugging=0 is convenient)
#   * run as:
#         $ ./anisoreductions.sh > anisoresiduals.txt
#   * generates additional file transcript.txt
#   * python script to post process in p4pdes-book/figs/anisoreductions.py

LEV=6    # 129x129 grid

TRANSCRIPT=transcript.txt

KSPITS=5

COMMON="-fsh_problem manupoly -fsh_initial_type random -da_refine $LEV -snes_type ksponly -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_monitor -ksp_max_it $KSPITS -ksp_rtol 0 -ksp_atol 0 -pc_type mg"

function runcase() {
  CMD="../fish $COMMON $1"
  echo $CMD >> $TRANSCRIPT
  rm -f tmp.txt
  /usr/bin/time -f "real %e" $CMD &> tmp.txt
  cat tmp.txt >> $TRANSCRIPT
  grep "0 KSP Residual norm" tmp.txt | awk 'NF>1{print $NF}'  # extract number only
  grep "$KSPITS KSP Residual norm" tmp.txt | awk 'NF>1{print $NF}'  # extract number only
  echo
  rm -f tmp.txt
}

rm -f $TRANSCRIPT

# compare Chebyshev, SSOR, GS smoothing with anisotropy
for SMOOTH in "" "-mg_levels_ksp_type richardson -mg_levels_pc_type sor" "-mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"; do
    echo "#SMOOTH = $SMOOTH"       # note "#" is comment character for numpy.loadtxt()
    for CY in 1.0 1.0e1 1.0e2 1.0e3; do
        runcase "-fsh_cy ${CY} $SMOOTH"
    done
done

