timing/
=======

Time results needed to make the text of the book are under git control in
this repo.  Thus they do not need to be re-generated to build the book.
That is, the book can be built without a working PETSc install.

To redo the timing results, for example for `tri.c` in Chapter 2, do

    $ (cd ../../c/ch2/ && make tri)
    $ cd tri/
    $ ./time.sh ../../../c/ch2/tri

The other cases are similar, though each script `time.sh` is special to the
example.

