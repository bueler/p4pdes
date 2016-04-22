Making movies
=============

The TS-using codes in this directory can generate binary files containing the _trajectory_, that is, the time-axis and the solution.  The book documents how to plot solutions as curves in the time-versus-solution plane, that is, as trajectories.  For the two codes that solve PDEs in two spatial dimensions, namely `heat.c` and `pattern.c`, one can also generate movies.

static trajectories
-------------------

Before doing movies, first we show how to view trajectories as curves.  One can use PETSc alone for a run-time "line-graph" view:

        make ode
        ./ode -ts_monitor_lg_solution draw -draw_pause 0.2

For saving an image there is a [python](https://www.python.org/)/[matplotlib](http://matplotlib.org/) script `plottrajectory.py` in this directory.  To save in file `result.png` do:

        ./ode -ts_monitor binary:t.dat -ts_monitor_solution binary:y.dat
        ./plottrajectory.py t.dat y.dat -o result.png

Note that

        ./plottrajectory.py t.dat y.dat

simply shows the result on the screen.

movies
------

If the PETSc code has two spatial dimensions, so that it uses both a `TS` and a 2D `DMDA` object, then viewing the whole trajectory requires a movie.  Codes `heat.c` and `pattern.c` in the current directory are of this type.

PETSc alone can generate a movie at run-time, for instance by

        make heat
        ./heat -da_refine 4 -ts_monitor_solution draw -draw_pause 0.2

The question is how to save a high-resolution movie for future viewing.  Here is an example.  First generate and save the solution in PETSc binary-format files:

        ./heat -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat \
            -ts_final_time 0.02 -ts_dt 0.001 -da_refine 6

Because this run reports that the grid has dimensions 193 by 192, we can show a [python](https://www.python.org/)/[matplotlib](http://matplotlib.org/)-generated movie on the screen this way:

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat

To save the frames in individual files:

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat -o bar

This generates files `bar000.png`, `bar001.png`, and so on, using the name pattern `bar%03d.png`.

From this collection of image files, the following commands use the [`ffmpeg`](https://www.ffmpeg.org/) tool to generate a `.m4v` format movie:

        ffmpeg -r 4 -i bar%03d.png bar.m4v

Of course one might need to install `ffmpeg`, so something like `sudo apt-get install ffmpeg` might be needed.  Viewing the movie might use `totem` or `vlc`.

The compression of the `.m4v` format is already substantial.  In particular, the result of

        ls -lh u.dat bar.m4v

is that `u.dat` is a 6 MB file and `bar.m4v` is a 44 KB file.

FIXME dof=2 for `pattern.c`
