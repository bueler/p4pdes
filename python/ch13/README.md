p4pdes/python/ch13/
===================

Only Chapter 13 of _PETSc for PDEs_ has [Python](https://www.python.org/)
example codes.  They use Python3 and [Firedrake](https://www.firedrakeproject.org/),
a finite element library based on PETSc data types and solvers.

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

        $ cd ch13/
        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ source ~/firedrake/bin/activate
        (firedrake) $ ./fish.py

Running the Stokes solver `stokes.py` is similar.

The author has set the following Bash alias for himself:

        alias drakeme='unset PETSC_DIR; unset PETSC_ARCH; source ~/firedrake/bin/activate'

### run the test suite

Do

        (firedrake) $ make test

### visualize results

The two codes `fish.py` and `stokes.py` allow the `-o foo.pvd` option which
writes a file which is readable with [Paraview](https://www.paraview.org/).

### cleaning up

Do

        $ make clean

