#!/bin/bash

# This script leaves only the "meat" of codes, for inclusion into the book.
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

# each of these filenames should be distinct
CH3="structuredpoisson.c poisson.c"
CH5="fish2.c"
CH7="readmesh.c poissontools.c poissonfem.c"
OTHER="obstacle.c"

mkdir cstrip/

for NAME in $CH3; do
    cp ../c/ch3/$NAME cstrip/$NAME
done
for NAME in $CH5; do
    cp ../c/ch5/$NAME cstrip/$NAME
done
for NAME in $CH7; do
    cp ../c/ch7/$NAME cstrip/$NAME
done
for NAME in $OTHER; do
    cp ../c/$NAME cstrip/$NAME
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
