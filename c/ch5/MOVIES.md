Making movies
=============

PETSc TS-using codes, like the ones in this directory, can generate binary files containing the _trajectory_, that is, the time-axis _t_ and the solution _y(t)_.  One may then plot solutions as curves in the time-versus-solution plane, as trajectories, and we show that first.

For the two codes that solve PDEs in two spatial dimensions, namely `heat.c` and `pattern.c`, however, one can visualize the full trajectory by generating a _movie_.  We show that kind of result second.

The method here, based on the [python](https://www.python.org/)/[matplotlib](http://matplotlib.org/) script `plotTS.py` in the current directory, and is light-weight but not at all full-featured.  This script also needs either local copies of, or sym-links to, `PetscBinaryIO.py` and `petsc_conf.py` which are in `$PETSC_DIR/bin/`.

For improved visualization one may either improve/modify the script or switch to a more advanced framework like [paraview](http://www.paraview.org/).


static trajectories
-------------------

First we show how to view trajectories as curves in the _t,y_ plane.  One gets multiple curves on the same graph if _y_ has dimension greater than one, as a PETSc Vec, which is the usual case.

One can use PETSc alone for a run-time "line-graph" view:

        make ode
        ./ode -ts_monitor_lg_solution -draw_pause 0.1

For saving an image file of type `.png` with the same basic appearance, use `plotTS.py`:

        ./ode -ts_monitor binary:t.dat -ts_monitor_solution binary:y.dat
        ./plotTS.py t.dat y.dat -o result.png

(A common error at this stage arises from not copying `PetscBinaryIO.py` and `petsc_conf.py` from `$PETSC_DIR/bin/` to the current directory, or making sym-links.)

Without option `-o` (or `-oroot` below), the script simply shows the result on the screen:

        ./plotTS.py t.dat y.dat

The heat equation solution from `heat.c` _can_ be viewed by one of the methods above, but it is not the natural and desired visualization.  Just give it a try to see!


movie for scalar PDE in spatial 2D
----------------------------------

Two of the codes here have two spatial dimensions, namely `heat.c` and `pattern.c`.  They use both a `TS` and a 2D `DMDA` object.  Viewing the trajectory they generate requires making a movie.

PETSc alone can generate a movie at run-time, for instance by

        make heat
        ./heat -da_refine 4 -ts_monitor_solution draw -draw_pause 0.2

The question is how to save a convenient, possibly high-resolution, movie for future viewing.  Here is an example.  First generate and save the solution in PETSc binary-format files:

        ./heat -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat \
            -ts_final_time 0.02 -ts_dt 0.001 -da_refine 5

This run reports that the grid has dimensions 193 by 192.  Adding these grid dimension as options, the same script shows a movie on the screen:

        ./plotTS.py -mx 129 -my 128 t.dat u.dat

Simply add a filename root to save the frames in individual files:

        ./plotTS.py -mx 129 -my 128 t.dat u.dat -oroot bar

This generates files `bar000.png`, `bar001.png`, and so on, using the name pattern `bar%03d.png`.

Now use the [`ffmpeg`](https://www.ffmpeg.org/) tool to generate a `.m4v` format movie from the collection of `.png` image files:

        ffmpeg -r 4 -i bar%03d.png bar.m4v     # set rate to 4 frames/second

Of course one might need to install `ffmpeg`, so something like `sudo apt-get install ffmpeg` might be needed.  Viewing the movie itself requires some viewer.  (On linux platforms, `totem` or `vlc` are possibilities.)

The compression from the `.m4v` format and `ffmpeg` is substantial.  In particular, `u.dat` is 6 MB while `bar.m4v` is only 44 KB.


movie for dof>1 PDE in spatial 2D
-----------------------------------

For problems with multiple degrees of freedom, like `pattern.c`, PETSc opens one window for each component:

        make pattern
        ./pattern -ts_adapt_type none -da_refine 4 -ts_final_time 300 -ts_dt 5 \
             -ts_monitor_solution draw

Generating the movie with `plotTS.py` requires setting the degrees of freedom `-dof 2`, and choosing one component using either option `-c 0` or `-c 1`.  For example:

        ./pattern -ts_adapt_type none -da_refine 5 -ts_final_time 300 -ts_dt 5 \
             -ts_monitor binary:t.dat -ts_monitor_solution binary:uv.dat
        ./plotTS.py -mx 96 -my 96 -dof 2 -c 0 t.dat uv.dat -oroot foo

Now we generate a movie the same way as above;

        ffmpeg -r 4 -i foo%03d.png foo.m4v

