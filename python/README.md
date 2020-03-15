p4pdes/python/
==============

The codes in Chapters 13 and 14 use [Firedrake](https://www.firedrakeproject.org/), a finite element library based on PETSc data types and solvers.  Firedrake uses [Python](https://www.python.org/) and [petsc4py](https://petsc4py.readthedocs.io/en/stable/).

These codes will remain here and be maintained and supported in the long term.

### configure and install Firedrake

To download and install Firedrake installation follow the instructions at the [download tab on the Firedrake page](https://www.firedrakeproject.org/download.html).  It is recommended that you allow Firedrake to download its own copy of PETSc.

After the initial download (of the Firedrake install script) do something like

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ python3 firedrake-install

Firedrake will then proceed to download and install its rather large stack of dependencies.  The reason to unset variables is so that Firedrake does its own PETSc install with its own compatible version of PETSc.

### start with the Poisson example

Do something like this to run the Poisson solver in Chapter 13, which will test your Firedrake installation:

        $ cd p4pdes/python/ch13/
        $ source ~/firedrake/bin/activate
        (firedrake) $ ./fish.py

Note that the default grid is 3 x 3 and the default KSP is CG.  Note the grid can be set by either `-refine X` or `-mx` and `-my`; using `-refine` allows geometric multigrid (GMG).  Solver options use prefix `-s_`.  Thus for CG+GMG on a fine grid do something like

        (firedrake) $ ./fish.py -refine 8 -s_pc_type mg -s_ksp_monitor

To see help do these; the first gives help specific to `fish.py` and the second gives the usual PETSc type of help (i.e. with all applicable PETSc options):

        (firedrake) $ ./fish.py -fishhelp
        (firedrake) $ ./fish.py -help

To test the Firedrake installation you can also do the following in either `ch13/` or `ch14/`:

        (firedrake) $ make test

