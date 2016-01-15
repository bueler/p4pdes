#!/bin/bash

# ./testit.sh PROGRAM OPTS PROCESSES TESTNUM

rm -f maketmp tmp difftmp

make $1 > maketmp 2>&1;

mpiexec -n $3 ./$1 $2 > tmp

diff output/$1.test$4 tmp > difftmp

CURRDIR=${PWD##*/}
if [[ -s difftmp ]] ; then
   echo "ERROR: Test #$4 of $CURRDIR/$1"
   echo "       on $3 processes FAILED; diffs follow:"
   cat difftmp
else
   echo "PASS: Test #$4 of $CURRDIR/$1"
   rm -f maketmp tmp difftmp
fi ;

