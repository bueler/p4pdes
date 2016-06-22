timing/
=======

This directory contains timing results needed for tables or graphs in the text of the book.  Because they do not need to be re-generated when the book is built from LaTeX sources, these timing results do not depend on a working PETSc installation.

Note that, by contrast, _convergence_ tables or graphs may need to be regenerated using PETSc.  For timing, however, I want to choose the machine.

The script `time.sh` in each subdirectory is special to the example.

example for a table
-------------------

To regenerate timing results for `tri.c` in Chapter 2, do

        $ (cd ../../c/ch2/ && export PETSC_ARCH=linux-c-opt && make tri)
        $ cd tri/
        $ ./time.sh ../../../c/ch2/tri

The results are then in files `timing/tri/X.Y` where `X` is a KSP choice and `Y` is a PC choice.  These files are read directly into the text of the book via LaTeX `\input` commands; see the `\intime` macro (re)defined when needed.

example for a timing figure
---------------------------

To regenerate timing (and other) results for `unfem.c` in Chapter 7, do

        $ (cd ../../c/ch7/ && export PETSC_ARCH=linux-c-opt && make unfem)
        $ cd unfem/
        $ ./time.sh ../../../c/ch7

