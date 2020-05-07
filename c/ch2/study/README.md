# timing results for a table in Chapter 2

For a matrix from `unfem.c` on a N=41409 node mesh go to `ch10/study/` and read comments in `genlinsys.sh`.  Make sure the PETSc build is `--with-debugging=0`.  Then return to the current directory and do

        $ ./time.sh "../loadsolve -fA ../../ch10/study/A.dat -fb ../../ch10/study/b.dat"

Watch run for convergence issues.  The richardson+jacobi runs diverge (DIVERGED_ITS).

