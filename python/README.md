p4pdes/python/
==============

FIXME  The codes in Chapters 1--12, i.e. in `ch*/`, are petsc4py versions of the corresponding codes in `c/ch*/`

The codes in Chapters 13 and 14 use [Firedrake](https://www.firedrakeproject.org/), a finite element library based on PETSc data types and solvers.  Firedrake uses [Python](https://www.python.org/) and [petsc4py](https://petsc4py.readthedocs.io/en/stable/).

These codes will remain here and be maintained and supported in the long term.

### configure and install Firedrake

To download and install Firedrake please follow the instructions at the [download tab on the Firedrake page](https://www.firedrakeproject.org/download.html).  It is recommended that you allow Firedrake to download and configure its own copy of PETSc.

After the initial download (of the Firedrake install script) do something like this:

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ python3 firedrake-install

Firedrake will then proceed to download and install its rather large stack of dependencies.  The reason to unset variables is so that Firedrake does its own PETSc install with its own compatible version of PETSc.

### getting started with the Poisson example

Do this to run the Poisson solver in Chapter 13, which will also test whether your Firedrake installation is working:

        $ cd p4pdes/python/ch13/
        $ source ~/firedrake/bin/activate
        (firedrake) $ ./fish.py

When you run a Firedrake program for the first time it will cache various finite element constructions.  Thus it will run much faster the second time.

Note that the default mesh is a 3 x 3 regular grid and the default KSP is CG.  Finer meshes can be set by either `-refine X` and/or `-mx` and `-my`.  Note that using `-refine` allows geometric multigrid (GMG).  PETSc solver options use prefix `-s_`.  Thus for CG+GMG on a fine grid with tight tolerances do something like this:

        (firedrake) $ ./fish.py -refine 8 -s_pc_type mg -s_ksp_monitor -s_ksp_rtol 1.0e-10

To see help do the following:

        (firedrake) $ ./fish.py -fishhelp
        (firedrake) $ ./fish.py -help

The first gives help specific to `fish.py`.  The second gives the usual PETSc type of help (i.e. with all applicable PETSc options); `grep` for specific PETSc options.

### software testing

To test the Firedrake installation you can also do the following in either `ch13/` or `ch14/`:

        (firedrake) $ make test

