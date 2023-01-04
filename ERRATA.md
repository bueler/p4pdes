# Errata for *PETSc for Partial Differential Equations*

Not everything in the book is perfect, so reader-submitted corrections and comments are very much appreciated.  Please submit them through the [issues](https://github.com/bueler/p4pdes/issues)  tab at the [github](https://github.com/bueler/p4pdes) site, or by email to the author at `elbueler@alaska.edu`.  Corrections to the example programs themselves will appear as commits in the repository, and then in the [releases](https://github.com/bueler/p4pdes/releases).

The list of errata below shows corrections to the text of the 2nd printing.  That is, these errors exist in both the 1st and 2nd printing.  See [errata for the 1st printing](ERRATA-1stprinting.md) for a list of errors which have been corrected in the 2nd printing.

### Chapter 6

* Page 131: The text after equation (6.4) incorrectly defines triangular matrices.  It should say "where d_{ij} = 0 if i \ne j, l_{ij} = 0 if i < j, and u_{ij} = 0 if i > j."

### Chapter 9

* Page 230: Formulas (9.25), (9.26), and (9.27) are all missing a "u" term.  There should be "+u \psi_{pq}" added to the integrand.  The code `phelm.c` itself is correct; see the `IntegrandRef()` function.
