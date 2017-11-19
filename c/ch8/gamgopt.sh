#!/bin/bash

# run "make test" and "./gensquares.sh" first
# see also gamgopt.sh to generate input files sq100.xxx, ..., sq800.xxx

#FIXME option for MAXLEV
#FIXME check if this works for Koch levels
for LEV in 1 2 3 4 5 6; do
    cmd="./unfem -un_case 3 -snes_type ksponly -ksp_rtol 1.0e-8 -snes_monitor_short -ksp_converged_reason -pc_type gamg -un_mesh meshes/sq${LEV}"
    rm -f foo.txt
    $cmd -log_view &> foo.txt
    'grep' "Linear solve converged due to" foo.txt
    'grep' "result for N" foo.txt
    'grep' "Time (sec):" foo.txt | awk '{print $3}'
done
rm -f foo.txt

