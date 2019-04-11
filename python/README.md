p4pdes/python/
==============

The codes in Chapter 13 use [Firedrake](https://www.firedrakeproject.org/), a finite element library based on PETSc data types and solvers.  Firedrake uses [Python](https://www.python.org/) and [petsc4py](https://petsc4py.readthedocs.io/en/stable/).

These codes will remain here and be maintained and supported in the long term.

### configure and install Firedrake

To download and install Firedrake installation follow the instructions at the [download tab on the Firedrake page](https://www.firedrakeproject.org/download.html).  It is recommended that Firedrake downloads its own copy of PETSc.

After the initial download (of the Firedrake install script) do something like

        $ unset PYTHONPATH; unset PETSC_DIR; unset PETSC_ARCH;
        $ python3 firedrake-install

The reason to unset variables is so that Firedrake does its own PETSc install with a version of PETSc that will work for it.

### run the Poisson example

Do something like this to run the Poisson solver:

        $ cd p4pdes/python/ch13/
        $ source ~/firedrake/bin/activate
        (firedrake) $ ./fish.py

Use `./fish.py --help` to get some options.

The default grid is 3 x 3 and the default KSP is CG.  Note the grid can be set by either `-refine X` or `-mx` and `-my`; using `-refine` allows geometric multigrid (GMG).  Solver options use prefix `-s_`.  Thus for CG+GMG on a fine grid do something like

        (firedrake) $ ./fish.py -refine 8 -s_pc_type mg -s_ksp_monitor

To get an output file in Paraview-readable form use `-o NAME.pvd`.

### run the Stokes example

Running the Stokes solver `stokes.py` is similar.  For a uniform grid with default Taylor-Hood elements do

        (firedrake) $ ./stokes.py -mx 65 -my 65 -s_ksp_monitor

Generally the options available for `fish.py` are also available for `stokes.py`, but see `--help` output.

For a nonuniform grid, with refinement in the lower corners, use a special script to generate a `.geo` geometry-description file and then use [Gmsh](http://gmsh.info/) to generate a mesh readable by `stokes.py`:

        (firedrake) $ ./lidbox.py foo.geo
        (firedrake) $ gmsh -2 foo.geo
        (firedrake) $ ./stokes.py -i foo.msh -s_ksp_monitor

For more about solver options see the book text in Chapter 13.


### run the test suite

Do

        (firedrake) $ make test

### visualize results

The two codes `fish.py` and `stokes.py` allow the `-o foo.pvd` option which writes a file which is readable with [Paraview](https://www.paraview.org/).

### cleaning up

Do

        $ make clean

