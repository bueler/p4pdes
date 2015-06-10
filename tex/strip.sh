#!/bin/bash

# This script leaves only the "meat" of codes, for inclusion into the book.
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

CH3="structuredpoisson.c poisson.c"
CH4="readmesh.c poissontools.c poisson.c"
CH5="poisson.c"
OTHER="obstacle.c"

for NAME in $CH3; do
    cp ../c/ch3/$NAME cstrip/ch3$NAME
done
for NAME in $CH4; do
    cp ../c/ch4/$NAME cstrip/ch4$NAME
done
for NAME in $CH5; do
    cp ../c/ch5/$NAME cstrip/ch5$NAME
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
