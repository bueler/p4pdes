#!/bin/bash

# This script leaves only the "meat" of codes, for inclusion into the book.
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

# filenames should be distinct
CH2="vecmatksp.c tri.c"
CH3="poisson.c"
CH4="expcircle.c ecjacobian.c reaction.c"
CH5="plap.c"
CH7="ad3.c"
CH10="readmesh.c poissontools.c poissonfem.c"
CH13="obstacle.c"

mkdir cstrip/

for NAME in $CH2; do
    cp ../c/ch2/$NAME cstrip/$NAME
done
for NAME in $CH3; do
    cp ../c/ch3/$NAME cstrip/$NAME
done
for NAME in $CH4; do
    cp ../c/ch4/$NAME cstrip/$NAME
done
for NAME in $CH5; do
    cp ../c/ch5/$NAME cstrip/$NAME
done
for NAME in $CH7; do
    cp ../c/ch7/$NAME cstrip/$NAME
done
for NAME in $CH10; do
    cp ../c/ch10/$NAME cstrip/$NAME
done
for NAME in $CH13; do
    cp ../c/ch13/$NAME cstrip/$NAME
done

#ls cstrip/
cd cstrip/

set -x

for NAME in *.c; do
    sed -i.bak 's/ierr = //g' $NAME
    sed -i.bak 's/CHKERRQ(ierr);//g' $NAME
    sed -i.bak '/\/\/STRIP/d' $NAME
    sed -i.bak '/PetscErrorCode ierr/d' $NAME
done
rm *.bak

