Hardware topology: process placement for performance
----------------------------------------------------

Telling your MPI to bind and/or map processes to your hardware will improve performance.  See the "Maximizing memory bandwidth" section of the [PETSc User's Manual](https://www.mcs.anl.gov/petsc/documentation/index.html).  First, a good idea is to install [`hwloc`](https://www.open-mpi.org/projects/hwloc/), including the program [`lstopo`](https://www.open-mpi.org/projects/hwloc/lstopo/) which can display your machine's "hardware topology".  (A `hwloc` package may be available from your package manager.)  Then do

        $ lstopo

to get a graphical view of the layout of sockets, cores, threads, and cache memory on your system.

### Multisocket example

For multisocket compute nodes, consider this example using a code from Chapter 6.  Make sure to use a `--with-debugging=0` PETSc configuration, and to start do `cd c/ch6/ && make fish`.  Then compare timing of these two runs:

        $ mpiexec -n P ./fish -da_refine L -pc_type mg -pc_mg_levels L
        $ mpiexec -n P --map-by socket --bind-to core ./fish -da_refine L -pc_type mg -pc_mg_levels L

Generate the timing by adding `-log_view |grep "Time (sec):"`.  Here `P` is at most the number of physical cores on your node and `L` is large enough to give many seconds of run time, but small enough to fit in memory (and your patience).  For example, try `L=9`.  For larger `P` values you may need to set ` -pc_mg_levels L-1` or ` -pc_mg_levels L-2` to further-reduce the depth of the multigrid (V) cycles so that parallel DMDA-based multigrid can solve the coarse grid problem.  (See Chapters 6--8 regarding parallel multigrid.)

An alternative mapping/binding is `--map-by core --bind-to hwthread` as in the next example.  In any case experimentation is in order.

### Single socket example

For a single socket machine with `P` or fewer cores, e.g. a laptop, and if each core supports multiple hyperthreads, here is a recommended setting for performance:

        $ mpiexec -n P --map-by core --bind-to hwthread ./fish -da_refine L -pc_type mg -pc_mg_levels L

The author gets a factor-of-two speedup on his laptop, with `P=4` runs on a four-physical-core processor, over the defaults.

### More information

_Why_ does such mapping and binding improve performance?  Also, _why_ are these not default settings for `mpiexec`?  We offer no clear justification, but read the [PETSc User's Manual](https://www.mcs.anl.gov/petsc/documentation/index.html).  Also see why processor affinity can effectively reduce cache problems at the [processor affinity wikipedia page](https://en.wikipedia.org/wiki/Processor_affinity).  See also [MPICH Hydra usage](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager) or the [Open-MPI mpirun man page](https://www.open-mpi.org/doc/current/man1/mpirun.1.php).

