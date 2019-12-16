#!/bin/bash

# A script to run regression tests from the c/chN/ directories.  Use "make test"
# which runs this script as follows:
#    ./testit.sh PROGRAM OPTS PROCESSES TESTNUM

rm -f maketmp tmp difftmp

make $1 > maketmp 2>&1;

grep warning maketmp

CURRDIR=${PWD##*/}

if [ $3 -eq 1 ]; then
    CMD="./$1 $2"
else
    CMD="mpiexec -n $3 ./$1 $2"
fi

if [[ ! -f output/$1.test$4 ]]; then
    echo "FAIL: Test #$4 of $CURRDIR/$1"
    echo "       command = '$CMD'"
    echo "       OUTPUT MISSING"

else

    $CMD > tmp

    diff output/$1.test$4 tmp > difftmp

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

