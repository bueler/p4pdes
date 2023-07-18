Hardware topology: process placement for performance
----------------------------------------------------

Telling your MPI process manager to bind and/or map processes to your hardware will improve performance.  See the ["Hints for Performance Tuning" section of the PETSc/TAO User's Manual](https://petsc.org/release/docs/manual/performance/).

### Understanding your hardware

First, a good idea is to install [`hwloc`](https://www.open-mpi.org/projects/hwloc/), including the program [`lstopo`](https://www.open-mpi.org/projects/hwloc/lstopo/) which can display your machine's "hardware topology".  (A `hwloc` package may be available from your package manager.)  Then do

        $ lstopo

to get a graphical view of the layout of sockets, cores, threads, and cache memory on your system.

Next, try running the [streams benchmark](https://www.cs.virginia.edu/stream/ref.html).  This can be done with or without process-placement options:

        $ cd $PETSC_DIR
        $ export PETSC_ARCH=linux-c-opt   # use a --with-debugging=0 build
        $ make streams
        $ make streams MPI_BINDING="-map-by numa -bind-to core"

The results typically suggest that a memory-bandwidth-limited computation will already saturate the memory bandwidth even for small numbers of processes; see the ["Hints" Manual section](https://petsc.org/release/docs/manual/performance/).  While [streams](https://www.cs.virginia.edu/stream/ref.html) does almost no computation compared to its memory transfers, even the numerical solution of a PDE is often memory-bandwidth-limited.

### Multisocket example

For multisocket compute nodes, consider this example using a code from Chapter 6.  Then compare timing of these two runs:

        $ cd ~/p4pdes/c/ch6/
        $ export PETSC_ARCH=linux-c-opt   # use a --with-debugging=0 build
        $ make fish
        $ mpiexec -n P ./fish -da_refine L -pc_type mg -pc_mg_levels L
        $ mpiexec -n P -map-by socket -bind-to core ./fish -da_refine L -pc_type mg -pc_mg_levels L

Generate the timing by adding `-log_view |grep "Time (sec):"`.  Here `P` is at most the number of physical cores on your node and `L` is large enough to give many seconds of run time, but small enough to fit in memory (and your patience).  For example, try `L=9`.  For larger `P` values you may need to set ` -pc_mg_levels L-1` or ` -pc_mg_levels L-2` to further-reduce the depth of the multigrid (V) cycles so that parallel DMDA-based multigrid can solve the coarse grid problem.  (See Chapters 6--8 regarding parallel multigrid.)

Alternative mapping/bindings are:
  * `-map-by numa -bind-to core`
  * `-map-by core -bind-to hwthread`
Experimentation is in order.

### Single node example

For a single node, multi-socket machine (e.g. some workstations), here is a recommended setting for performance:

        $ mpiexec -n P -map-by numa -bind-to core ./fish -da_refine L -pc_type mg -pc_mg_levels L

### More information

_Why_ does such mapping and binding improve performance?  Also, _why_ are these not default settings for `mpiexec`?  Read the [PETSc/TAO User's Manual](https://petsc.org/release/docs/manual/), but also see why processor affinity can effectively reduce cache problems at the [processor affinity wikipedia page](https://en.wikipedia.org/wiki/Processor_affinity).  See also [MPICH Developer Documentation](https://github.com/pmodels/mpich/blob/main/doc/wiki/Index.md) or the [Open-MPI mpirun man page](https://www.open-mpi.org/doc/current/man1/mpirun.1.php).
