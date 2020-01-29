Hardware topology, and binding for performance
==============================================

Installing [`hwloc`](https://www.open-mpi.org/projects/hwloc/), which includes the program [`lstopo`](https://www.open-mpi.org/projects/hwloc/lstopo/), is a good idea; see [`www.open-mpi.org/projects/hwloc`](https://www.open-mpi.org/projects/hwloc/).  A `hwloc` package may be available from your package manager.

Then do

        $ lstopo

to get a graphical view of the layout of sockets, cores, threads, and caches on your system.  This is your "hardware topology".

It seems that binding processes to cores improves performance.  For example, compare timing, of runs like

        $ cd c/ch6/ && make fish
        $ mpiexec -n P ./fish -da_refine L -pc_mg -pc_mg_levels L
        $ mpiexec -n P --bind-to core ./fish -da_refine L -pc_mg -pc_mg_levels L

where `P` is equal to the number of physical cores on your node and `L` is large enough to give many seconds of run time, but small enough to fit in memory.  (You may need to set ` -pc_mg_levels L-1` or ` -pc_mg_levels L-2` to make sure parallel DMDA-based multigrid can solve the coarse grid problem in parallel.  Also, use a `--with-debugging=0` PETSc configuration.  See Chapters 6, 7, and 8 of the book for explanations of these parenthetical comments.)

_Why_ does such binding improve performance?  Read about why processor affinity can effectively reduce cache problems at (for example) the [processor affinity wikipedia page](https://en.wikipedia.org/wiki/Processor_affinity).  See also [MPICH Hydra usage](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager) or the [Open-MPI mpirun man page](https://www.open-mpi.org/doc/current/man1/mpirun.1.php).

