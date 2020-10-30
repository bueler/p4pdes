Errata for "PETSc for PDEs"
---------------------------

Not everything in the book is correct, so reader corrections and comments are much appreciated!  Please submit corrections and issues through the [Issues](https://github.com/bueler/p4pdes/issues) or [Pull requests](https://github.com/bueler/p4pdes/pulls) tabs.

This list of errata shows corrections to the text of the published book.  Corrections to the example programs (i.e. in `c/` and `py/` in the current repo) will appear as commits here and thus will appear in the [releases](https://github.com/bueler/p4pdes/releases) over time.

* Page 3: In `e.c`, a better way to handle an error code returned from `PetscInitialize()` is to get its output `ierr` and then add the check `if (ierr) return ierr;`  The example programs now do this, but remember that error-checking is stripped from the displayed examples later in the book (e.g. for `vecmatksp.c` on page 28).

* Page 15: In the footnote: "solver" --> "solvers".

* Page 29: In the first complete sentence on the page, for clarity add "the action of" before M^{-1}.

* Page 47: In the fourth sentence of the text on this page, substitute "global" for "local".  (This is the worst error found so far!  Can you beat it?)

* Page XXX: NEXT?

