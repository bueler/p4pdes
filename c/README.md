p4pdes/c/
=========

This directory contains the C programs which support the book _PETSc for PDEs_.

  * I will maintain and support these examples in the long term and update
    and tag them with PETSc versions.

### install PETSc

Please follow the instructions at [petsc.org/release/install/](https://petsc.org/release/install/).

Notes:

  * My book does not contain PETSc installation instructions, though it has
    minimal advice.  Maintaining installation information is the job of
    PETSc developers, not me.  (Thank goodness!)
  * Please make sure to [download the latest release of p4pdes](https://github.com/bueler/p4pdes/releases/) which is compatible with your PETSc version.  Note that package managers (such as [apt for debian systems](https://wiki.debian.org/Apt)) may contain older versions of PETSc that are incompatible with the newest release of [p4pdes](https://github.com/bueler/p4pdes/).
  * [CONFIGS.md](CONFIGS.md) contains some of the `configure` commands which
    work on the author's machines.  These are minimal installation suggestions.
  * See [`../README.md`](../README.md) for how to set-up shell environments so that you can go back and forth between the PETSc version you use for C program development, and also maintain a copy of PETSc which was installed to support Firedrake.
  * My book does not help the reader with tools for debugging C programs, but that is an important skill to have.

### compile and run one example

Do this to build and run the program `fish.c` from Chapter 6, which solves the Poisson equation.  This solves in parallel on a 1025 x 1025 grid in a couple of seconds:

    $ cd ch6
    $ make fish
    $ mpiexec -n 4 ./fish -pc_type mg -da_refine 9 -ksp_monitor

### software (regression) testing

    $ make test           # in either c/ or c/ch*/

### cleaning up

    $ make clean          # in either c/ or c/ch*/
