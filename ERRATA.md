# Errata for *PETSc for Partial Differential Equations*

Not everything in the book is perfect, so reader-submitted corrections and comments are very much appreciated.  Please submit them through the [issues](https://github.com/bueler/p4pdes/issues) or [pull requests](https://github.com/bueler/p4pdes/pulls) tabs at the [github](https://github.com/bueler/p4pdes) site, or by email to the author at `elbueler@alaska.edu`.  Corrections to the example programs themselves will appear as commits in the repository, and then in the [releases](https://github.com/bueler/p4pdes/releases).

The list of errata below shows corrections to the text, including to the 2nd printing.

### Chapter 9

* Page 230: Formulas (9.25), (9.26), and (9.27) are all missing the "u" term.  There should be "+u \psi_{pq}" added to the integrand.  (The code `phelm.c` is correct; see the `IntegrandRef()` function.)
