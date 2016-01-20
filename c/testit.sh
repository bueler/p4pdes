#!/bin/bash

# ./testit.sh PROGRAM OPTS PROCESSES TESTNUM

rm -f maketmp tmp difftmp

make $1 > maketmp 2>&1;

CMD="mpiexec -n $3 ./$1 $2"

if [[ ! -f output/$1.test$4 ]]; then
    echo "FAIL: Test #$4 of $CURRDIR/$1"
    echo "       command = '$CMD'"
    echo "       OUTPUT MISSING"

else

    $CMD > tmp

    diff output/$1.test$4 tmp > difftmp

    CURRDIR=${PWD##*/}
    if [[ -s difftmp ]] ; then
       echo "FAIL: Test #$4 of $CURRDIR/$1"
       echo "       command = '$CMD'"
       echo "       diffs follow:"
       cat difftmp
    else
       echo "PASS: Test #$4 of $CURRDIR/$1"
       rm -f maketmp tmp difftmp
    fi

fi

