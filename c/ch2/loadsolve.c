static char help[] =
"Load a matrix  A  and right-hand-side  b  from a binary file (PETSc format).\n"
"Solve the system  A x = b  and provide timing of KSPSolve in seconds.\n"
"Example.  First save a system from tri.c:\n"
"  ./tri -ksp_view_mat binary:Ab.dat -ksp_view_rhs binary:Ab.dat::append\n"
"then load it and time the solution:\n"
"  ./loadsolve -f Ab.dat\n"
"(This is a simpler code than src/ksp/ksp/examples/tutorials/ex10.c.)\n";

/*
small system example:
./tri -ksp_view_mat binary:Ab.dat -ksp_view_rhs binary:Ab.dat::append
./loadsolve -f Ab.dat -ksp_view_mat -ksp_view_rhs -ksp_view_solution

small system example where RHS missing:
./tri -ksp_view_mat binary:A.dat
./loadsolve -f A.dat -norhs -ksp_view_mat -ksp_view_rhs

large tridiagonal system (m=10^7) example:
./tri -tri_m 10000000 -ksp_view_mat binary:large.dat -ksp_view_rhs binary:large.dat::append
./loadsolve -f large.dat

use of PetscTime() and PetscTimeSubtract() seems to produce same result as -log_view:
./loadsolve -f large.dat -log_view &> foo
grep KSPSolve foo
head -n 1 foo
*/

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Vec         x, b;
  Mat         A;
  KSP         ksp;
  PetscBool   flg,
              norhs = PETSC_FALSE,
              notime = PETSC_FALSE,
              verbose = PETSC_FALSE;
  char        file[PETSC_MAX_PATH_LEN] = "";     /* input file name */
  PetscViewer lsfile;

  PetscInitialize(&argc,&args,NULL,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for loadsolve",""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f","input file containing linear system (A,b)",
                            "loadsolve.c",file,file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-norhs","do not try to load right-hand-side b from file (use zero instead)",
                          "loadsolve.c",norhs,&norhs,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-notime","do not report timing of KSPSolve",
                          "loadsolve.c",notime,&notime,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-verbose","say what is going on",
                          "loadsolve.c",verbose,&verbose,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!flg || strlen(file)==0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "no input file ... ending  (usage: loadsolve -f file.dat)\n"); CHKERRQ(ierr);
      return 1;
  }

  if (verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "reading linear system from %s ...\n",file); CHKERRQ(ierr);
  }
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&lsfile);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,lsfile);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (norhs || VecLoad(b,lsfile)) {
      int m;
      ierr = MatGetSize(A,&m,NULL); CHKERRQ(ierr);
      if (verbose) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,
             "right-hand-side b missing from input file ... using zero vector of length %d\n",m); CHKERRQ(ierr);
      }
      ierr = VecSetSizes(b,PETSC_DECIDE,m); CHKERRQ(ierr);
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&lsfile);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  if (!notime) {
      PetscLogDouble solvetime;
      ierr = PetscTime(&solvetime);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);
      ierr = PetscTimeSubtract(&solvetime);CHKERRQ(ierr);
      if (verbose) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,
                 "PetscTime says KSPSolve took this many seconds:\n"); CHKERRQ(ierr);
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%f\n",-solvetime); CHKERRQ(ierr);
  } else {
      ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);
  }

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}

