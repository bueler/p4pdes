static char help[] = "Assemble a Mat sparsely.\n";

#include <petsc.h>

int main(int argc,char **args) {
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

  PetscCall(PetscInitialize(&argc,&args,NULL,help));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetValues(A,3,i1,3,j1,aA1,INSERT_VALUES));
  PetscCall(MatSetValues(A,1,&i2,3,j2,aA2,INSERT_VALUES));
  PetscCall(MatSetValue(A,i3,j3,aA3,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}
