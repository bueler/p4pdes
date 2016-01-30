static char help[] = "Solve a 4x4 linear system using Vec, Mat, and KSP.\n"
"Assemble the Mat sparsely.\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Vec    x, b;
  Mat    A;
  KSP    ksp;
  int    jb[4] = {0, 1, 2, 3},
         i1[3] = {0, 1, 2},
         j1[3] = {0, 1, 2},
         i2 = 3,
         j2[3] = {1, 2, 3},
         i3 = 1,
         j3 = 3;
  double ab[4]  = {7.0, 1.0, 1.0, 3.0},
         aA1[9] = { 1.0,  2.0,  3.0,
                    2.0,  1.0, -2.0,
                   -1.0,  1.0,  1.0},
         aA2[3] = { 1.0,  1.0, -1.0},
         aA3 = -3.0;

  PetscInitialize(&argc,&args,NULL,help);

  ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,4); CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  ierr = VecSetValues(b,4,jb,ab,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);
  ierr = MatSetValues(A,3,i1,3,j1,aA1,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i2,3,j2,aA2,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(A,i3,j3,aA3,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}

