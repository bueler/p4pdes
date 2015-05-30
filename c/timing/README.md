p4pdes/c/timing/
================

Time results needed to make the text of the book are under git control in
this repo.  Thus they do not need to be generated to build the book, so that
the book can be built without a working PETSc install.

To redo the timing results do

    $ (cd ../ && make c1tri)
    $ cd c1tri/
    $ ./time.sh

To remove all timing results, do

    $ cd c1tri/
    $ rm *.*.*

