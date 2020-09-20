Errata for "PETSc for PDEs"
---------------------------

It is unlikely that everything in the book is correct, so reader corrections and comments are much appreciated!  Please submit corrections and issues through the [Issues](https://github.com/bueler/p4pdes/issues) or [Pull requests](https://github.com/bueler/p4pdes/pulls) tabs.

This list of errata shows corrections to the text of the published book.  Corrections to the example programs (i.e. in `c/` and `py/` in the current repo) are addressed through git and will appear [in the releases](https://github.com/bueler/p4pdes/releases).

* Page 3: In `e.c`, better way to handle the possibility of an error from `PetscInitialize()` is to get its output `ierr` and then add the check `if (ierr) return ierr;`  The example programs now _all_ do this, but remember that error-checking is stripped from the displayed examples later in the book (e.g. for `vecmatksp.c` on page 28).

* Page XXX: NEXT

