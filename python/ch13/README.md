p4pdes/python/ch13
==================

These codes are examples in Chapter 13 of _PETSc for PDEs_.  They use
[Python3](https://www.python.org/) and [Firedrake](https://www.firedrakeproject.org/).
Firedrake is a finite element library which uses PETSc as a solver, and which
uses PETSc DMPlex to manage unstructured meshes.

These examples will remain here and be maintained and supported in the long
term.

### configure and install Firedrake

A Firedrake installation follows the instructions at 

### run a code

Do something like this to run the Poisson solver:

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ source ~/firedrake/bin/activate
        $ ./firefish.py

FIXME

The Stokes solver is similar.

### visualize the results

FIXME Paraview

### cleaning up

    $ make clean

