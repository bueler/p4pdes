Running jobs on a cluster
=========================

This subdirectory contains scripts, including an example batch script `cluster.sh`,
which runs codes from the book _PETSc for Partial Differential Equations_ on a
Linux cluster.  The example which might be useful as a suggestion for how
PETSc codes are run in a batch system on many-node machines, but users should
expect to make modifications for their own machines.

The particular settings are for a modest-sized Linux cluster at the University
of Alaska Fairbanks; see http://www.gi.alaska.edu/research-computing-systems/hpc/chinook.
Access is generally limited to UAF students and faculty.

The script `genweak.py` generates batch scripts suitable for the weak-scaling
study in Chapter 8 of the book.
