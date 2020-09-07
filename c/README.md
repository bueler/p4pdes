p4pdes/c/
=========

This directory contains C programs which support the book _PETSc for PDEs_.

  * I will maintain and support these examples in the long term and update
    and tag them with PETSc versions.

### install PETSc

Follow instructions at [www.mcs.anl.gov/petsc/documentation/installation.html](http://www.mcs.anl.gov/petsc/documentation/installation.html).

  * My book does not contain PETSc installation instructions, though it has
    minimal advice.  Maintaining installation information is the job of
    PETSc developers, not me.  (Thank goodness.)
  * Also, my book does not help the reader with debugging C programs.
  * However, [CONFIGS.md](CONFIGS.md) contains some of the `configure`
    commands which work on the author's machines.  These are minimal
    installation examples/suggestions.

### compile and run one example

Do this to build and run the program `e.c` in Chapter 1:

    $ cd ch1
    $ make e
    $ ./e
    $ mpiexec -n 20 ./e

### regression testing

    $ make test               # in either c/ or c/ch*/

### cleaning up

    $ make distclean          # in either c/ or c/ch*/

