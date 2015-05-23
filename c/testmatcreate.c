
static char help[] =
"Test MatCreate() and MatSetValues(), for Chapter 1 example.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

Mat A;
PetscInt  i, j[4] = {0, 1, 2, 3};
PetscReal v[4];

MatCreate(PETSC_COMM_WORLD,&A);
MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4);
MatSetFromOptions(A);
MatSetUp(A);

i = 0;  v[0] = 1.0;  v[1] = 2.0;  v[2] = 3.0;  v[3] = 4.0;
MatSetValues(A,1,&i,4,j,v,INSERT_VALUES);
i = 1;  v[0] = 2.0;  v[1] = 0.0;  v[2] = -2.0;  v[3] = -3.0;
MatSetValues(A,1,&i,4,j,v,INSERT_VALUES);
i = 2;  v[0] = -1.0;  v[1] = 1.0;  v[2] = 1.0;  v[3] = 0.0;
MatSetValues(A,1,&i,4,j,v,INSERT_VALUES);
i = 3;  v[0] = 2.0;  v[1] = 1.0;  v[2] = -1.0;  v[3] = 1.0;
MatSetValues(A,1,&i,4,j,v,INSERT_VALUES);

MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  ierr = MatDestroy(&A); CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
//END
