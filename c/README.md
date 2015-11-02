p4pdes/c/
=========

Codes to support a book on using PETSc for PDEs.

  * I will maintain and support these example codes in the long term.

### install PETSc

Follow instructions at [www.mcs.anl.gov/petsc/documentation/installation.html](http://www.mcs.anl.gov/petsc/documentation/installation.html).

  * My book does not contain installation instructions or advice.  Also, it does not help the reader with debugging C programs.  Maintaining such information is the job of PETSc developers, not me.  Thank goodness.

### compile and run

Do this to build the first code:

    $ cd ch1
    $ make e
    $ ./e
    $ mpiexec -n 20 ./e

Tested with PETSc maint branch (at 3.6.2).

