p4pdes
======

_PETSc for Partial Differential Equations_ is a book on using [PETSc](https://petsc.org/release/) and [Firedrake](https://www.firedrakeproject.org/) to solve [partial differential equations](https://en.wikipedia.org/wiki/Partial_differential_equation) by modern numerical methods.

<p align="center">
  <a  href="https://doi.org/10.1137/1.9781611976311"> <img src="frontcover.jpg" alt="image of front cover" /img> </a>
</p>

Order a paper copy from [SIAM Press](https://doi.org/10.1137/1.9781611976311), or the e-book from [Google Play](https://play.google.com/store/books/details/Ed_Bueler_PETSc_for_Partial_Differential_Equations?id=tgMHEAAAQBAJ).

This repository contains the C and Python example programs upon which the book is based.

**These example programs will remain here for the long term, and they will be maintained for future versions of PETSc.**

### C examples

To compile and run the C examples, for Chapters 1 through 12, see the [`README.md`](c/README.md) in the `c/` directory.

### Python/Firedrake examples

Chapters 13 and 14 use [Firedrake](https://www.firedrakeproject.org/), a [Python](https://www.python.org/) finite element library based on PETSc.  See the [`README.md`](python/README.md) in the `python/` directory to run these examples.

### Spring 2025 update on managing two PETSc installations:

Running all the codes from the book can be done with two copies of PETSc.  One copy is any PETSc installation, to be used for the C codes in Chapters 1--12; this one can be updated to follow any branch of PETSc, for example.  The other copy is separate, and configured so that Firedrake works; this one is for the Python codes in Chapters 13 & 14.

Note that, as of March 2025, downloading and installing Firedrake, as in the instructions at the [Install tab on the Firedrake page](https://www.firedrakeproject.org/install.html), is usually done by building a copy of PETSc from source, using Firedrake's recommended configuration flags, and then installing Firedrake via [pip](https://pypi.org/project/pip/).

To install and manage these two PETSc copies I do the following:

  1. I configure and build one copy with any preferred flags, supporting my development of C programs:
```
  git clone -b release https://gitlab.com/petsc/petsc.git petsc
```
  Most configuration choices will be compatible with building and running the codes in Chapters 1--12.  Note that Fortran support is not needed for the book's codes.

  2. I follow the instructions at the [Install tab on the Firedrake page](https://www.firedrakeproject.org/install.html) to install Firedrake.  However, I do this inside a directory `Firedrake` so that the second PETSc copy needed by Firedrake is in a different location:
```
  mkdir Firedrake
  cd Firedrake/
  git clone --depth 1 https://github.com/firedrakeproject/petsc.git petsc
```
  I do the installation, also from within `Firedrake/`, making sure that `PETSC_DIR` points to `Firedrake/petsc/`.  For example, in my case I see the environment variables:
```
   CC=mpicc CXX=mpicxx PETSC_DIR=/home/bueler/Firedrake/petsc PETSC_ARCH=arch-firedrake-default HDF5_MPI=ON
```
  Then the stages of starting the virtual environment and doing `pip install ...` go on as documented at the [Firedrake installation page](https://www.firedrakeproject.org/install.html).

  3. Finally, I add certain convenience functions to `.bashrc` in my home directory:
```
parse_git_dirty() {
    [[ $(git status 2> /dev/null | tail -n1) != "nothing to commit, working tree clean" ]] && echo "*"
}
parse_git_branch() {
    git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e "s/* \(.*\)/[\1$(parse_git_dirty)]/"
}
petscme() {
    export PETSC_DIR=~/petsc;
    export PETSC_ARCH=linux-c-dbg;
    alias mpiexec=$PETSC_DIR/$PETSC_ARCH/bin/mpiexec;
    export PS1='(petsc) \[\033[0;33m\]\w\[\033[0m\]$(parse_git_branch)$ '
}
drakeme() {
    source ~/Firedrake/venv-firedrake/bin/activate
    export CC=mpicc CXX=mpicxx PETSC_DIR=~/Firedrake/petsc PETSC_ARCH=arch-firedrake-default HDF5_MPI=ON
}
```
  These posix-compatible Bash functions provide informative prompts for what mode I am in, and also what Git branch I am on.  For example, here is how I start to work with the C codes in Chapters 1--12:
```
  ~/p4pdes/c[master]$ petscme
  (petsc) ~/p4pdes/c[master]$
```
  For working with the Firedrake Python codes in Chapters 13 & 14, I do:
```
  ~/p4pdes/python[master]$ drakeme
  (venv-firedrake) ~/p4pdes/python[master]$
```
