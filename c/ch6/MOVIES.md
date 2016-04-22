Making movies
=============

PETSc TS-using codes, like the ones in this directory, can generate binary files containing the _trajectory_, that is, the time-axis _t_ and the solution _y(t)_.  One may then plot solutions as curves in the time-versus-solution plane, as trajectories, and we show that first.

For the two codes that solve PDEs in two spatial dimensions, namely `heat.c` and `pattern.c`, however, one can visualize the full trajectory by generating a _movie_.  We show that kind of result second.

The method here, based on the [python](https://www.python.org/)/[matplotlib](http://matplotlib.org/) script `plottrajectory.py` in the current directory, and is light-weight but not at all full-featured.  This script also needs either local copies of, or sym-links to, `PetscBinaryIO.py` and `petsc_conf.py` which are in `$PETSC_DIR/bin/`.

For improved visualization one may either improve/modify the script or switch to a more advanced framework like [paraview](http://www.paraview.org/).


static trajectories
-------------------

First we show how to view trajectories as curves in the _t,y_ plane.  One gets multiple curves on the same graph if _y_ has dimension greater than one, as a PETSc Vec, which is the usual case.

One can use PETSc alone for a run-time "line-graph" view:

        make ode
        ./ode -ts_monitor_lg_solution draw -draw_pause 0.2

For saving an image file of type `.png` with the same basic appearance, use `plottrajectory.py`:

        ./ode -ts_monitor binary:t.dat -ts_monitor_solution binary:y.dat
        ./plottrajectory.py t.dat y.dat -o result.png

A common error at this stage comes from the script not having access to `PetscBinaryIO.py` and `petsc_conf.py` from `$PETSC_DIR/bin/`.

Note that without option `-o` (or `-oroot` below), the script simply shows the result on the screen:

        ./plottrajectory.py t.dat y.dat

The heat equation solution from `heat.c` _can_ be viewed by one of the methods above, but it is not the natural and desired visualization.  Just give it a try to see!


movie for scalar PDE in spatial 2D
----------------------------------

If the PETSc code has two spatial dimensions, so that it uses both a `TS` and a 2D `DMDA` object, then viewing the whole trajectory requires a movie.  Codes `heat.c` and `pattern.c` in the current directory are of this type.

PETSc alone can generate a movie at run-time, for instance by

        make heat
        ./heat -da_refine 4 -ts_monitor_solution draw -draw_pause 0.2

The question is how to save a convenient, possibly high-resolution, movie for future viewing.  Here is an example.  First generate and save the solution in PETSc binary-format files:

        ./heat -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat \
            -ts_final_time 0.02 -ts_dt 0.001 -da_refine 6

This run reports that the grid has dimensions 193 by 192.  Adding these grid dimension as options, the same script shows a movie on the screen:

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat

Simply add a filename root to save the frames in individual files:

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat -oroot bar

This generates files `bar000.png`, `bar001.png`, and so on, using the name pattern `bar%03d.png`.

From this collection of image files, the following commands use the [`ffmpeg`](https://www.ffmpeg.org/) tool to generate a `.m4v` format movie:

        ffmpeg -r 4 -i bar%03d.png bar.m4v

Of course one might need to install `ffmpeg`, so something like `sudo apt-get install ffmpeg` might be needed.  Viewing the movie itself requires some viewer; `totem` or `vlc` are possibilities.

The compression from the `.m4v` format and `ffmpeg` is already substantial.  In particular, the result of

        ls -lh u.dat bar.m4v

is that `u.dat` is a 6 MB file and `bar.m4v` is a 44 KB file.


movie for dof>1 PDE in spatial 2D
-----------------------------------

For problems with multiple degrees of freedom, like `pattern.c`, PETSc opens one window for each component:

        make pattern
        ./pattern -ts_adapt_type none -da_refine 4 -ts_final_time 300 -ts_dt 5 \
             -ts_monitor_solution draw

Generating the movie with `plottrajectory.py` requires setting the degrees of freedom `-dof 2`, and choosing one component using either option `-c 0` or `-c 1`.  For example:

        ./pattern -ts_adapt_type none -da_refine 5 -ts_final_time 300 -ts_dt 5 \
             -ts_monitor binary:t.dat -ts_monitor_solution binary:uv.dat
        ./plottrajectory.py -mx 96 -my 96 -dof 2 -c 0 t.dat uv.dat -oroot foo

Now we generate a movie the same way as above;

        ffmpeg -r 4 -i foo%03d.png foo.m4v

