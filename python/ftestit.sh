#!/bin/bash

# A script to run regression tests for Firedrake-based programs
# from the python/chN/ directories.  Use "make test"
# which runs this script as follows:
#    ./ftestit.sh PROGRAM OPTS PROCESSES TESTNUM
# Compare c/testit.sh.

rm -f maketmp tmp difftmp

make $1 > maketmp 2>&1;

grep warning maketmp

CURRDIR=${PWD##*/}

FAIL="FAIL"
if [ $3 = "INFO" ]; then
    CMD="$1 $2"
    FAIL="INFO"
elif [ $3 -eq 1 ]; then
    CMD="./$1 $2"
else
    CMD="mpiexec -n $3 ./$1 $2"
fi

if [[ ! -f output/$1.test$4 ]]; then
    echo "FAIL: Test #$4 of $CURRDIR/$1"
    echo "       command = $CMD"
    echo "       OUTPUT MISSING"

else

    $CMD &> tmp

    diff output/$1.test$4 tmp > difftmp

    if [[ -s difftmp ]] ; then
       echo "$FAIL: Test #$4 of $CURRDIR/$1"
       echo "       command = $CMD"
       echo "       diffs follow:"
       cat difftmp
    else
       echo "PASS: Test #$4 of $CURRDIR/$1"
       rm -f maketmp tmp difftmp
    fi

fi

