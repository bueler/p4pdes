static char help[] =
"Load a matrix  A  and right-hand-side  b  from binary files (PETSc format).\n"
"Then solve the system  A x = b  using KSPSolve().\n"
"Example.  First save a system from tri.c:\n"
"  ./tri -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat\n"
"then load it and solve:\n"
"  ./loadsolve -fA A.dat -fb b.dat\n"
"To time the solution read the third printed number:\n"
"  ./loadsolve -fA A.dat -fb b.dat -log_view |grep KSPSolve\n"
"(This is a simpler code than src/ksp/ksp/examples/tutorials/ex10.c.)\n";

/*
small system example w/o RHS:
./tri -ksp_view_mat binary:A.dat
./loadsolve -fA A.dat -ksp_view_mat -ksp_view_rhs

large tridiagonal system (m=10^7) example:
./tri -tri_m 10000000 -ksp_view_mat binary:A.dat -ksp_view_rhs binary:b.dat
./loadsolve -fA A.dat -fb b.dat -log_view |grep KSPSolve
*/

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Vec         x, b;
  Mat         A;
  KSP         ksp;
  PetscInt    m, n, mb;
  PetscBool   flg,
              verbose = PETSC_FALSE;
  char        nameA[PETSC_MAX_PATH_LEN] = "",
              nameb[PETSC_MAX_PATH_LEN] = "";
  PetscViewer fileA, fileb;

  ierr = PetscInitialize(&argc,&args,NULL,help); if (ierr) return ierr;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for loadsolve",""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-fA","input file containing matrix A",
                            "loadsolve.c",nameA,nameA,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-fb","input file containing vector b",
                            "loadsolve.c",nameb,nameb,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-verbose","say what is going on",
                          "loadsolve.c",verbose,&verbose,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (strlen(nameA) == 0) {
      SETERRQ(PETSC_COMM_SELF,1,
              "no input matrix provided ... ending  (usage: loadsolve -fA A.dat)\n");
  }

  if (verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "reading matrix from %s ...\n",nameA); CHKERRQ(ierr);
  }
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,nameA,FILE_MODE_READ,&fileA);CHKERRQ(ierr);
  ierr = MatLoad(A,fileA);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fileA);CHKERRQ(ierr);
  ierr = MatGetSize(A,&m,&n); CHKERRQ(ierr);
  if (verbose) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "matrix has size m x n = %d x %d ...\n",m,n); CHKERRQ(ierr);
  }
  if (m != n) {
      SETERRQ(PETSC_COMM_SELF,2,"only works for square matrices\n");
  }

  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (flg) {
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,nameb,FILE_MODE_READ,&fileb);CHKERRQ(ierr);
      if (verbose) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,
             "reading vector from %s ...\n",nameb); CHKERRQ(ierr);
      }
      ierr = VecLoad(b,fileb);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fileb);CHKERRQ(ierr);
      ierr = VecGetSize(b,&mb); CHKERRQ(ierr);
      if (mb != m) {
          SETERRQ(PETSC_COMM_SELF,3,"size of matrix and vector do not match\n");
      }
  } else {
      if (verbose) {
          ierr = PetscPrintf(PETSC_COMM_WORLD,
             "right-hand-side vector b not provided ... using zero vector of length %d\n",m); CHKERRQ(ierr);
      }
      ierr = VecSetSizes(b,PETSC_DECIDE,m); CHKERRQ(ierr);
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
  }

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x); CHKERRQ(ierr);
  ierr = VecSet(x,0.0); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&b);
  return PetscFinalize();
}

