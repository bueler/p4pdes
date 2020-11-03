Errata for "PETSc for PDEs"
---------------------------

Not everything in the book is correct, and so reader corrections and comments are much appreciated!  Please submit corrections and issues through the [Issues](https://github.com/bueler/p4pdes/issues) or [pull requests](https://github.com/bueler/p4pdes/pulls) tabs.  Corrections to the example programs (i.e. files in `c/` and `py/`) will appear as commits in the current repo and then in the [releases](https://github.com/bueler/p4pdes/releases).

The list of errata below shows corrections to the text of the published book, including notable ones labeled **BAD ONE**.

* Page 3: In the program `e.c`, a better way to handle an error code returned from `PetscInitialize()` is to get its output `ierr` and then add the check `if (ierr) return ierr;`  The example programs now do this.

* Page 15: In the footnote: "solver" --> "solvers".

* Page 29: In the first complete sentence on the page, for clarity add "the action of" before M^{-1}.

* Page 47 **BAD ONE**: In the fourth sentence of the text on this page, substitute "global" for "local".

* Page 50: In the first sentence after Code 3.1, add "the" before "locally owned".

* Page 89: In the first sentence after Table 4.3, replace "both" with "either".

* Page 97: In the sentence defining the local truncation error, remove the first "O(h^1)".

* Page 106: In the first complete sentence on the page, replace "when" with "if and only if".

* Page 110: In the sentence starting "All of these", replace "latter two runs" with "first two runs".

* Page 132 **BAD ONE**: In equations (6.6) and (6.7) replace `u_k[i]` with `u_k[j]` and `u_{k+1}[i]` with `u_{k+1}[j]`.

