Making movies
=============

The TS-using codes in this directory can generate binary files containing the
time-axis and the solution.  The book documents how to plot solutions as curves
in the time-versus-solution plane, that is, as trajectories.  For the two codes
that solve PDEs in two spatial dimensions, namely `heat.c` and `pattern.c`, one
can also generate movies.

plotting trajectories
---------------------

FIXME

plotting movies
---------------

First note that it is easy to generate a movie at run-time, for instance by

        ./heat -da_refine 4 -ts_monitor_solution draw -draw_pause 0.2

So the question is, for example, how to save a high-resolution movie for future viewing.  Here is an example.  First generate and save the solution in PETSc
binary-format files:

        ./heat -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat \
            -ts_final_time 0.02 -ts_dt 0.001 -da_refine 6

Note that this run reports that the grid has dimensions 193 by 192.

To show this on the screen do:

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat

However, we want to save the frames and turn them into a movie in an easy-to
-distibute format.

        ./plottrajectory.py -mx 193 -my 192 t.dat u.dat -o bar

This generates files `bar000.png`, `bar001.png`, and so on, in the name
pattern `bar%03d.png`.  From this collection of image files, the following
commands use the [`ffmpeg`](https://www.ffmpeg.org/) tool to generate a
`.m4v` format movie:

        ffmpeg -r 4 -i bar%03d.png bar.m4v

Note that one might need to install `ffmpeg`, so something like `sudo apt-get install ffmpeg` might be needed.  Viewing the movie with a command like `totem var.m4v` or `vlc bar.m4v`.

FIXME note compression


