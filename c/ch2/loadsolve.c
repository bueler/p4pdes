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
  Vec         x, b;
  Mat         A;
  KSP         ksp;
  PetscInt    m, n, mb;
  PetscBool   flg,
              verbose = PETSC_FALSE;
  char        nameA[PETSC_MAX_PATH_LEN] = "",
              nameb[PETSC_MAX_PATH_LEN] = "";
  PetscViewer fileA, fileb;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));

  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for loadsolve","");
  PetscCall(PetscOptionsString("-fA","input file containing matrix A",
                               "loadsolve.c",nameA,nameA,PETSC_MAX_PATH_LEN,NULL));
  PetscCall(PetscOptionsString("-fb","input file containing vector b",
                               "loadsolve.c",nameb,nameb,PETSC_MAX_PATH_LEN,&flg));
  PetscCall(PetscOptionsBool("-verbose","say what is going on",
                             "loadsolve.c",verbose,&verbose,NULL));
  PetscOptionsEnd();

  if (strlen(nameA) == 0) {
      SETERRQ(PETSC_COMM_WORLD,1,
              "no input matrix provided ... ending  (usage: loadsolve -fA A.dat)\n");
  }

  if (verbose) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
         "reading matrix from %s ...\n",nameA));
  }
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,nameA,FILE_MODE_READ,&fileA));
  PetscCall(MatLoad(A,fileA));
  PetscCall(PetscViewerDestroy(&fileA));
  PetscCall(MatGetSize(A,&m,&n));
  if (verbose) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
         "matrix has size m x n = %d x %d ...\n",m,n));
  }
  if (m != n) {
      SETERRQ(PETSC_COMM_WORLD,2,"only works for square matrices\n");
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetFromOptions(b));
  if (flg) {
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,nameb,FILE_MODE_READ,&fileb));
      if (verbose) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
             "reading vector from %s ...\n",nameb));
      }
      PetscCall(VecLoad(b,fileb));
      PetscCall(PetscViewerDestroy(&fileb));
      PetscCall(VecGetSize(b,&mb));
      if (mb != m) {
          SETERRQ(PETSC_COMM_WORLD,3,"size of matrix and vector do not match\n");
      }
  } else {
      if (verbose) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD,
             "right-hand-side vector b not provided ... using zero vector of length %d\n",m));
      }
      PetscCall(VecSetSizes(b,PETSC_DECIDE,m));
      PetscCall(VecSet(b,0.0));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecSet(x,0.0));
  PetscCall(KSPSolve(ksp,b,x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}
