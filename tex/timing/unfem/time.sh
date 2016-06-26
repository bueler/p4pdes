#!/bin/bash

set +x

LOC=$1

function run() {
  rm -f tmp
  set -x
  $LOC/unfem -log_view -un_mesh $LOC/meshes/$1.$3 $4 &> tmp
  set +x
  grep "|u-u_ex|_inf" tmp | awk '{print $10}' >> $2     # h
  grep "|u-u_ex|_inf" tmp | awk '{print $NF}' >> $2     # error
  grep "SNESFunctionEval" tmp | awk '{print $2}' >> $2  # evals
  grep "Read mesh      :" tmp | awk '{print $5}' >> $2  # stage time
  grep "Set-up         :" tmp | awk '{print $4}' >> $2  # stage time
  grep "Solver         :" tmp | awk '{print $4}' >> $2  # stage time
  rm -f tmp
}

# case 0
NAME=trap
(cd $LOC/ && ./gentraps.sh $NAME 9)
OUT=snes-fd-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5; do
    run $NAME $OUT $N "-snes_fd"
done
OUT=ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5 6 7 8 9; do
    run $NAME $OUT $N
done

#case 1
export OUT=case1-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5 6 7 8 9; do
    run $NAME $OUT $N "-un_case 1"
done

#case 2
NAME=trapneu
(cd $LOC/ && ./gentraps.sh $NAME 9)
export OUT=case2-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5 6 7 8 9; do
    run $NAME $OUT $N "-un_case 2"
done

