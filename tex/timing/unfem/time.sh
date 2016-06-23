#!/bin/bash

set +x

LOC=$1

function run() {
  rm -f tmp
  set -x
  $LOC/unfem -log_view -un_mesh $LOC/meshes/trap.$1 $2 &> tmp
  set +x
  grep "|u-u_exact|" tmp | awk '{print $10}' >> $OUT      # h
  grep "|u-u_exact|" tmp | awk '{print $NF}' >> $OUT      # error
  grep "SNESFunctionEval" tmp | awk '{print $2}' >> $OUT  # evals
  grep "Time (sec):" tmp | awk '{print $3}' >> $OUT       # time
  rm tmp
}

(cd $LOC/ && ./gentraps.sh trap 8)

export OUT=snes-fd-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5; do
    run $N "-snes_fd"
done

export OUT=ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5 6 7 8; do
    run $N
done

