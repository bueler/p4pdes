Hardware topology, and process placement for performance
========================================================

A good idea is to install [`hwloc`](https://www.open-mpi.org/projects/hwloc/), including the program [`lstopo`](https://www.open-mpi.org/projects/hwloc/lstopo/) which can display your machine's "hardware topology".  (A `hwloc` package may be available from your package manager.)

Do

        $ lstopo

to get a graphical view of the layout of sockets, cores, threads, and caches on your system.

Telling your MPI to bind and/or map processes to your hardware will improve performance.  See the "Maximizing memory bandwidth" section of the [PETSc User's Manual](https://www.mcs.anl.gov/petsc/documentation/index.html).

For multisocket compute nodes, consider this example using a code from Chapter 6.  Make sure to use a `--with-debugging=0` PETSc configuration, and to start do `cd c/ch6/ && make fish`.  Then compare timing of these two runs:

        $ mpiexec -n P ./fish -da_refine L -pc_type mg -pc_mg_levels L
        $ mpiexec -n P --bind-to core --map-by socket ./fish -da_refine L -pc_type mg -pc_mg_levels L

Generate the timing by adding `-log_view |grep "Time (sec):"`.

Here `P` is at most the number of physical cores on your node and `L` is large enough to give many seconds of run time, but small enough to fit in memory and your patience; try `L=9,10` for example.  For larger `P` values you may need to set ` -pc_mg_levels L-1` or ` -pc_mg_levels L-2` to make sure parallel DMDA-based multigrid can solve the coarse grid problem; see Chapters 6--8.  An alternative process placement is to try `--bind-to hwthread`.

_Why_ does such binding improve performance?  Read the [PETSc User's Manual](https://www.mcs.anl.gov/petsc/documentation/index.html).  Also see why processor affinity can effectively reduce cache problems at the [processor affinity wikipedia page](https://en.wikipedia.org/wiki/Processor_affinity).  See also [MPICH Hydra usage](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager) or the [Open-MPI mpirun man page](https://www.open-mpi.org/doc/current/man1/mpirun.1.php).

