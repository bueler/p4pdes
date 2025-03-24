p4pdes/python/
==============

The codes in Chapters 13 and 14 use [Firedrake](https://www.firedrakeproject.org/), a finite element library based on PETSc data types and solvers.  Firedrake uses [Python](https://www.python.org/) and [petsc4py](https://petsc4py.readthedocs.io/en/stable/).

These codes will remain here and be maintained and supported in the long term.

### configure and install Firedrake

To download and install Firedrake please follow the instructions at the [Install tab on the Firedrake page](https://www.firedrakeproject.org/install.html).  As of March 2025 this is usually done by building a copy of PETSc from source, using Firedrake's recommended configuration flags, and then installing Firedrake via [pip](https://pypi.org/project/pip/).

See the [top-level README](../README.md) for how to set-up shell environment variables so that you can go back and forth between the PETSc version you use for C program developement, and the copy of PETSc which was installed to support Firedrake.

### getting started with the Poisson example

Do this to run the Poisson solver in Chapter 13, which will also test whether your Firedrake installation is working:

        $ cd p4pdes/python/ch13/
        $ [activate your Firedrake venv]
        (firedrake) $ ./fish.py

When you run a Firedrake program for the first time it will cache various finite element constructions.  Thus it will run much faster the second time.

Note that the default mesh is a 3 x 3 regular grid and the default KSP is CG.  Finer meshes can be set by either `-refine X` and/or `-mx` and `-my`.  Note that using `-refine` allows geometric multigrid (GMG).  PETSc solver options use prefix `-s_`.  Thus for CG+GMG on a fine grid with tight tolerances do something like this:

        (firedrake) $ ./fish.py -refine 8 -s_pc_type mg -s_ksp_monitor -s_ksp_rtol 1.0e-10

To see help do the following:

        (firedrake) $ ./fish.py -fishhelp
        (firedrake) $ ./fish.py -help

The first gives help specific to `fish.py`.  The second gives the usual PETSc type of help, i.e. with all of the many applicable PETSc options.  Do

        (firedrake) $ ./fish.py -help |grep STRING

to find specific PETSc options including `STRING`.

### software testing

To test the Firedrake installation you can also do the following in either `ch13/` or `ch14/`:

        (firedrake) $ make test
