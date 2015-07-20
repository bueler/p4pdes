#!/bin/bash

# This script leaves only the "meat" of codes, for inclusion into the book.
# We remove "PetscErrorCode ierr" lines and "//STRIP" lines and
# characters "ierr = " and "CHKERRQ(ierr);"

# filenames should be distinct
CH3="structuredpoisson.c poisson.c"
CH4="expcircle.c expcircleJAC.c reaction.c"
CH6="fish2.c"
CH8="readmesh.c poissontools.c poissonfem.c"
OTHER="obstacle.c"

mkdir cstrip/

for NAME in $CH3; do
    cp ../c/ch3/$NAME cstrip/$NAME
done
for NAME in $CH4; do
    cp ../c/ch4/$NAME cstrip/$NAME
done
for NAME in $CH6; do
    cp ../c/ch6/$NAME cstrip/$NAME
done
for NAME in $CH8; do
    cp ../c/ch8/$NAME cstrip/$NAME
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
