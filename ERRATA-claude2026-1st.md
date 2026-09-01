# More errata for *PETSc for Partial Differential Equations*

This list was produced by an automated read-through by Claude Opus 5 in September 2026,
of a PDF of the 1st printing.  The list was then pruned and edited by the author.
Items already listed in `ERRATA.md` or `ERRATA-1stprinting.md` were excluded.

Method: text extracted with `pdftotext -layout`; page mapping verified against two known
errata entries.  Displayed equations and code listings extract poorly and are therefore
essentially unchecked, apart from the Chapter 3 exponent below.  A few items near the end
are style-consistency observations rather than clear errors.

### Chapter 2

* Page 18: "whether p_{n-1} is a "good" for approximately inverting A" should drop "a".

* Page 18: "is a thus a question of approximation theory" has a doubled "a".

* Page 31: "a LU factorization algorithm" should be "an LU".

### Chapter 3

* Page 56: "as expected from a O(h^2) FD" should be "an O(h^2)".

* Page 62: "apparently kappa_2(A_h) = O(h^2) for our FD scheme" should be O(h^{-2}).

### Chapter 4

* Page 72: "It may be come as a surprise" should be "It may come as a surprise".

* Page 76: "MatSetValues(), is also used the same way." has a stray comma.

* Page 86: "for a N = 8193 point grid" should be "an N = 8193".

* Page 87: "where mf in stands for "matrix-free."" has a stray "in".

* Page 91: "Thus a local minima of phi" should be "a local minimum".

### Chapter 5

* Page 109: "may be needed to solver the nonlinear equations" should be "solve".

* Page 113: In the Figure 5.5 caption, "the the method of lines" has a doubled "the".

* Page 113: In Code 5.5 the comment reads `// default to hx=hx=0.25 grid`; should be `hx=hy`.

* Page 115: "using a O(h_x^2) centered finite difference approximation" should be "an O(".

* Page 120: "visualizations shows large, though bounded, oscillations" should be "visualizations show".

* Page 123: "let us to refer to values of u and v" should be "let us refer".

### Chapter 6

* Page 130: "(How is the grid is partitioned into subdomains?" has a doubled "is".

* Page 130: "A common case is that A_pre is sparse matrix Mat object" is missing "a" before "sparse".

* Page 134: "smooth the error. though they are slow to eliminate it."  The period should be a comma.

* Page 140: "though the boundary value is indeterminant" should be "indeterminate".

* Page 148: In the Figure 6.8 caption, "Subgrid decompositions of a 8x8 grid" should be "an 8x8".

* Page 153: In footnote 25, "The documentation for PCMGSetRestriction() and PCMGSetInterpolation() say" should be "says".

* Page 158: "Note that grids can by viewed at run time" should be "can be viewed".

* Page 161: "a Cheybshev smoother" should be "Chebyshev".

### Interlude

* Page 173: "then use values q.n, q.xi[r], ..." should be "and then use values ...".

### Chapter 7

* Page 177: "in each case N, the total number degrees of freedom, spans" is missing "of" before "degrees".

* Page 190: in "programmer effort to implement an analytical Jacobian implementation," delete "implementation".

TODO WIP: Ed needs to check from here

### Chapter 8

* Pages 204-215: The notation for parallel speedup is inconsistent.  The Definition on page 204 introduces lowercase `s_N(P)` (matching `e_N(P)` for efficiency), but capital `S_N(P)` appears on pages 204, 205, 206 (three times), and 215.

* Page 208: "on a 8193 x 8193 grid" should be "an 8193 x 8193".

* Page 212: "DM objects are designed, to, among other things, facilitate scalable PCSetup stages" has a stray comma after "designed".

* Page 213: "multigrid is essential for optimality (Chapters 6)" should be "(Chapter 6)".  This is the only such singular/plural slip in the book.

* Page 215: In Exercise 8.1(c), "Choose one the several PDE solutions" is missing "of".

### Chapter 10

* Page 251: ""1 2" says this a one-dimensional segment" is missing "is".

* Page 251: "include both the original values using in the .geo file" should be "used in".

* Page 258: "patterns ... show symmetry, and shows the diagonal entries" should be "and show".

* Page 258: "as the evaluations exceeds the default -snes_max_funcs" should be "exceed".

* Page 263: "shows that former run issues about 4 x 10^4 mallocs" is missing "the" before "former".

* Page 267: In the Figure 10.14 caption, "define overlapping aggreggates" should be "aggregates".

* Page 270: In the Figure 10.18 caption, "the the Picard iteration" has a doubled "the".

* Page 271: "Now we generate a N = 1025^2 approx 10^6 grid" should be "an N =".

* Page 272: "(i) reading an mesh from PETSc binary files" should be "a mesh".

* Page 274: "A common way to add (i)-(iii) is to use a FE discretization library" should be "an FE".

* Page 277: In Exercise 10.12, "Re-write c/ch9/koch/domain.py" should be **c/ch10/koch/domain.py**; page 272 gives the correct path, and the file is at `c/ch10/koch/domain.py` in the repository.

### Chapter 11

* Page 296: "because at a smooth local extrema the limiter imposes" -- "a ... extrema" should be "a ... extremum".  (This is separate from the existing errata item about "limiter imposes" in the same sentence.)

* Page 300: "solves (11.30) by a FV method similar to the one in advect.c" should be "an FV".

### Chapter 12

* Page 315: "on a 2D structured DMDA grids" should be "on 2D structured DMDA grids".

* Page 323: "Why is the SNES count is increasing?" has a doubled "is".

### Chapter 13

* Page 331: "FEnICs" appears three times on this page; elsewhere the book spells it "FEniCS" (four occurrences).

* Page 332: "which define a FE function space" should be "an FE".

* Page 336: "faster than any negative power of the k" should be "power of k".

### Chapter 14

* Page 344: "To compute the divergence a tensor, such as div sigma in (14.1)" is missing "of".

* Page 345: "the dimensionless Reynold's number R" should be "Reynolds number".

* Page 350: "(Recall that running Firedrake require activation of the Python virtual environment" should be "requires".

* Page 364: "This smoother change, and imperfect algorithmic scaling (Figures 14.7 and 14.8) accounts for the small increase" should be "account for".

### Index

* Page 384: "Lax-Richtmeyer equivalence theorem, 58" should be "Lax-Richtmyer".  The text on page 58 spells it correctly.

### Capitalization consistency

The book writes "in Chapter N" and "in Table N" (74 and 31 occurrences) when naming a
numbered chapter or table, but the following use a capital where the word is a common noun:

* Page xi: "each Chapter includes a demonstration"
* Page 36: "matrices from tri.c in this Chapter (left)"
* Page 297: "The cases in the Table with one iteration"
* Page 319: "Our code in this Chapter, obstacle.c"
* Page 331: "specific to this Chapter's examples"

(Note "this Chapter 2 example" on page 167 and "the Chapter 7 definition" on page 336 name
specific chapters and are correct as printed.)
