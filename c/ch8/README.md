c/ch8/
======

This directory contains scripts which runs codes from other Chapters of the book
_PETSc for Partial Differential Equations_.

These scripts demonstrate the use of a batch system on a Linux cluster, and they
support the discussion of parallel scaling in Chapter 8.  The scripts might be
useful as a suggestion for using a batch system on other clusters, but
users should expect to make modifications for their own machines.

The particular settings are for a modest-sized cluster `chinook` at the
University of Alaska Fairbanks; see
   https://www.gi.alaska.edu/services/research-computing-systems.
Access to `chinook` is generally limited to UAF students and faculty.

First see the example batch script `cluster.sh` which has a variety of medium-
scale runs from various Chapters in the book.

There are also [Python]() scripts which generate the data for certain figures
in Chapter 8 of the book:

  * `genstrong.py` generates batch scripts for the strong-scaling study using
    `ch7/minimal.c`; this data is fit to estimate serial fraction in Amdahl's law
  * `genweak.py` generates batch scripts for the weak-scaling study using BOTH
    `ch7/minimal.c` and `ch6/fish.c`

