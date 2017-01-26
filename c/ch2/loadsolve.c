static char help[] =
"Load a matrix and right-hand-side from a binary file (PETSc format): A x = b.\n"
"Solve the system and provide timing information.\n"
"For example, save a system from tri.c:\n"
"  ./tri -ksp_view_mat binary:Ab.dat -ksp_view_rhs binary:Ab.dat::append\n"
"then load and solve it\n"
"  ./loadsolve -f Ab.dat\n\n";

/*
extended small system example:
./tri -ksp_view_mat binary:Ab.dat -ksp_view_rhs binary:Ab.dat::append
./loadsolve -f Ab.dat -ksp_view_mat -ksp_view_rhs -ksp_view_solution

extended small system example where RHS missing:
./tri -ksp_view_mat binary:A.dat
./loadsolve -f A.dat -norhs -ksp_view_mat -ksp_view_rhs
*/

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Vec         x, b;
  Mat         A;
  KSP         ksp;
  PetscBool   flg, norhs=PETSC_FALSE;
  char        file[PETSC_MAX_PATH_LEN] = "";     /* input file name */
  PetscViewer lsfile;              /* viewer */

  PetscInitialize(&argc,&args,NULL,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for loadsolve",""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f","input file containing linear system (A,b)",
                            "loadsolve.c",file,file,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-norhs","do not try to load right-hand-side b from file (use zero instead)",
                          "loadsolve.c",norhs,&norhs,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  if (!flg) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "no input file ... ending  (usage: loadsolve -f file.dat)\n"); CHKERRQ(ierr);
      return 0;
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,
     "reading linear system from %s ...\n",file); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&lsfile);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatLoad(A,lsfile);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  if (norhs || VecLoad(b,lsfile)) {
      int m;
      ierr = MatGetSize(A,&m,NULL); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
         "right-hand-side b missing from input file ... using zero vector of length %d\n",m); CHKERRQ(ierr);
      ierr = VecSetSizes(b,PETSC_DECIDE,m); CHKERRQ(ierr);
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&lsfile);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);

// FIXME time solution

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}

