CONFIGS.md
==========

This is a list of recent configure commands, defining different `PETSC_ARCH` values, on the author's machine.

  * `linux-c-dbg`:

        ./configure --download-mpich --download-hypre --download-zlib --download-p4est --with-debugging=1

  * `linux-c-opt`:

        ./configure --download-mpich --download-hypre --download-zlib --download-p4est --with-debugging=0

  * `linux-c-quad`:

        ./configure --download-mpich --with-64-bit-indices --with-precision=__float128 --download-f2cblaslapack --with-debugging=1
