p4pdes/python/
==============

Only Chapter 13 of _PETSc for PDEs_ has Python example codes.  They use
[Python3](https://www.python.org/) and [Firedrake](https://www.firedrakeproject.org/).
Firedrake is a finite element library which uses PETSc DMPlex to manage
unstructured meshes and PETSc solvers.

These examples will remain here and be maintained and supported in the long
term.

### configure and install Firedrake

To download and install Firedrake installation follow the instructions at the
[download tab on the Firedrake page](https://www.firedrakeproject.org/download.html).
It is recommended that Firedrake downloads its own copy of PETSc, so after
downloading the Firedrake install script do something like

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ python3 firedrake-install

### run a code

Do something like this to run the Poisson solver:

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ source ~/firedrake/bin/activate
        $ cd ch13/
        $ ./firefish.py

Running the Stokes solver is similar.

### visualize the results

The `ch13/` codes allow the `-o foo.pvd` to write a file which is readable with
[Paraview](https://www.paraview.org/).

### cleaning up

        $ cd ch13/
        $ make clean

