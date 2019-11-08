static char help[] = "Assemble a Mat sparsely.\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode ierr;
  Mat        A;
  PetscInt   i1[3] = {0, 1, 2},
             j1[3] = {0, 1, 2},
             i2 = 3,
             j2[3] = {1, 2, 3},
             i3 = 1,
             j3 = 3;
  PetscReal  aA1[9] = { 1.0,  2.0,  3.0,
                        2.0,  1.0, -2.0,
                       -1.0,  1.0,  1.0},
             aA2[3] = { 1.0,  1.0, -1.0},
             aA3 = -3.0;

  PetscInitialize(&argc,&args,NULL,help);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);
  ierr = MatSetValues(A,3,i1,3,j1,aA1,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&i2,3,j2,aA2,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValue(A,i3,j3,aA3,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  MatDestroy(&A);
  return PetscFinalize();
}

