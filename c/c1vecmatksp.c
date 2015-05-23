static char help[] = "Solve a 4x4 linear system with Vec, Mat, and KSP.\n";

#include <petsc.h>

int main(int argc,char **args) {
  Vec            x,b;
  Mat            A;
  KSP            ksp;
  PetscInt       ib[4] = {0, 1, 2, 3},
                 i, j[3];
  PetscReal      vb[4] = {11.0, 7.0, 5.0, 3.0},
                 v[2]  = {4.0, -1.0}, w[3] = {-1.0, 4.0, -1.0};
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,4); CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x); CHKERRQ(ierr);

  ierr = VecSetValues(b,4,ib,vb,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  i = 0;  j[0] = 0;  j[1] = 1;
  ierr = MatSetValues(A,1,&i,2,j,v,INSERT_VALUES); CHKERRQ(ierr);
  i = 1;  j[0] = 0;  j[1] = 1;  j[2] = 2;
  ierr = MatSetValues(A,1,&i,3,j,w,INSERT_VALUES); CHKERRQ(ierr);
  i = 2;  j[0] = 1;  j[1] = 2;  j[2] = 3;
  ierr = MatSetValues(A,1,&i,3,j,w,INSERT_VALUES); CHKERRQ(ierr);
  i = 3;  j[0] = 2;  j[1] = 3;
  ierr = MatSetValues(A,1,&i,2,j,w,INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
