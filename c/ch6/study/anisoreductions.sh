#!/bin/bash
set -e

# for anisotropic form of Poisson in 2D, with random initial, comparing
# Chebyshev, SSOR, and GS smoothers, measure reduction in norm from KSPITS
# V-cycles

#   * timings do not matter (but using --with-debugging=0 is convenient)
#   * run as:
#         $ ./anisoreductions.sh > anisoresiduals.txt
#   * python script to post process in p4pdes-book/figs/anisoreductions.py
#   * generates additional files trans_*.txt, post-processed by
#     p4pdes-book/figs/anisostagnate.py

LEV=7    # 257x257 grid

TRANSCRIPT=transcript.txt

COMMON="-fsh_problem manupoly -fsh_initial_type random -da_refine $LEV -ksp_type richardson -ksp_norm_type unpreconditioned -ksp_monitor -ksp_rtol 0 -ksp_atol 0 -pc_type mg"

function runcasetranscripttmp() {
  CMD="../fish $COMMON $1"
  echo $CMD >> $2
  rm -f tmp.txt
  /usr/bin/time -f "real %e" $CMD &> tmp.txt
  cat tmp.txt >> $2
}

KSPITS=5

function runcase() {
  runcasetranscripttmp "${1}" transcript.txt
  grep "0 KSP Residual norm" tmp.txt | awk 'NF>1{print $NF}'  # extract number only
  grep "$KSPITS KSP Residual norm" tmp.txt | awk 'NF>1{print $NF}'  # extract number only
  echo
}

rm -f $TRANSCRIPT

# compare Chebyshev+SSOR, Chebyshev+ICC, Richardson+SSOR, Richardson+GS smoothing
# (and different levels of anisotropy)
for SMOOTH in "" "-mg_levels_pc_type icc" "-mg_levels_ksp_type richardson -mg_levels_pc_type sor" "-mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward"; do
    echo "#SMOOTH = $SMOOTH"       # note "#" is comment character for numpy.loadtxt()
    for CY in 1.0 1.0e1 1.0e2 1.0e3; do
        runcase "-ksp_max_it ${KSPITS} -fsh_cy ${CY} ${SMOOTH}"
        rm -f tmp.txt
    done
done


# put longer runs into individual transcripts for stagnation figure
# (which is created via c/sneskspplot.py)
KSPITS=12
CY=1.0e3
runcasetranscripttmp "-ksp_max_it ${KSPITS} -fsh_cy ${CY}" trans_chebssor.txt
rm -f tmp.txt
runcasetranscripttmp "-ksp_max_it ${KSPITS} -fsh_cy ${CY} -mg_levels_pc_type icc" trans_chebicc.txt
rm -f tmp.txt
runcasetranscripttmp "-ksp_max_it ${KSPITS} -fsh_cy ${CY} -mg_levels_ksp_type richardson -mg_levels_pc_type sor" trans_richssor.txt
rm -f tmp.txt
runcasetranscripttmp "-ksp_max_it ${KSPITS} -fsh_cy ${CY} -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward" trans_richgs.txt
rm -f tmp.txt

