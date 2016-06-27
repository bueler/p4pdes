#!/bin/bash

set +x

LOC=$1

function run() {
  rm -f tmp jetmp
  set -x
  $LOC/unfem -log_view -un_mesh $LOC/meshes/$1.$3 $4 &> tmp
  set +x
  grep "|u-u_ex|_inf" tmp | awk '{printf "%s ", $10}' >> $2     # h
  grep "|u-u_ex|_inf" tmp | awk '{printf "%s ", $NF}' >> $2     # error
  grep "SNESFunctionEval" tmp | awk '{printf "%s ", $2}' >> $2  # eval (count)
  grep "Read mesh      :" tmp | awk '{printf "%s ", $5}' >> $2  # stage time (s)
  grep "Set-up         :" tmp | awk '{printf "%s ", $4}' >> $2  # stage time
  grep "Solver         :" tmp | awk '{printf "%s ", $4}' >> $2  # stage time
  grep "Residual eval  :" tmp | awk '{printf "%s ", $5}' >> $2  # stage time
  grep "Jacobian eval  :" tmp > jetmp
  if [ -s jetmp ]  # if has contents
  then
      cat jetmp | awk '{printf "%s ", $5}' >> $2          # stage time
  else
      echo "Jacobian eval line EMPTY"
  fi
  echo >> $2
  #grep "Jacobian eval  :" tmp | awk '{print $5}' >> $2          # stage time (and newline)
  rm -f tmp jetmp
}

# case 0
NAME=trap
(cd $LOC/ && ./gentraps.sh $NAME 9)
OUT=snes-fd-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5; do
    run $NAME $OUT $N "-snes_fd"
done
OUT=snes-mf-ErrorsEvalsTimes
rm -f $OUT
for N in 1 2 3 4 5 6 7; do
    run $NAME $OUT $N "-snes_mf"
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

