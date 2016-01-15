#!/bin/bash

# ./testit.sh PROGRAM OPTS PROCESSES TESTNUM

# if passes: silent
# if fails: shows diffs

rm -f maketmp tmp difftmp

make $1 > maketmp 2>&1;

mpiexec -n $3 ./$1 $2 > tmp

diff output/$1.test$4 tmp > difftmp

if [[ -s difftmp ]] ; then
   echo "ERROR: test #$4 of $1 on $3 processes FAILED; diffs follow:"
   cat difftmp
else
   rm -f maketmp tmp difftmp
fi ;

