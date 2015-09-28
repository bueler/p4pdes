p4pdes/tex/
======

LaTeX sources for a book on using PETSc for PDEs.

Do this to build the PDF:

    $ make

For this to work, at a minimum you should have working versions of
* the `pdflatex` and `bibtex` commands from [LaTeX](https://www.latex-project.org/),
* the [`tufte-book`](https://tufte-latex.github.io/tufte-latex/) LaTeX class,
* [Python](https://www.python.org/) including [`scipy`](http://www.scipy.org/) and [`matplotlib`](http://matplotlib.org/) libraries,
* [`triangle`](https://www.cs.cmu.edu/~quake/triangle.html),
* and [`pdfcrop`](https://www.ctan.org/pkg/pdfcrop?lang=en).
