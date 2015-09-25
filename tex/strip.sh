#!/bin/bash

# This script leaves only the "meat" of codes, for inclusion into the book.
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

# filenames should be distinct
CH2="vecmatksp.c tri.c"
CH3="structuredpoisson.c poisson.c"
CH4="expcircle.c ecjacobian.c reaction.c"
CH5="plap.c"
CH6="fish2.c"
CH8="readmesh.c poissontools.c poissonfem.c"
CH11="obstacle.c"

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
for NAME in $CH6; do
    cp ../c/ch6/$NAME cstrip/$NAME
done
for NAME in $CH8; do
    cp ../c/ch8/$NAME cstrip/$NAME
done
for NAME in $CH11; do
    cp ../c/ch11/$NAME cstrip/$NAME
done

ls cstrip/
cd cstrip/

set -x

for NAME in *.c; do
    cp $NAME $NAME.tmp
    sed -i 's/ierr = //' $NAME.tmp
    sed -i 's/CHKERRQ(ierr);//' $NAME.tmp
    sed -i '/\/\/STRIP/d' $NAME.tmp
    sed -i '/PetscErrorCode ierr/d' $NAME.tmp
    mv $NAME.tmp $NAME
done

