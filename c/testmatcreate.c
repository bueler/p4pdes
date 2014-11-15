
static char help[] =
"Test MatCreate() and MatSetValues(), for Chapter 1 example.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

Mat A;
PetscInt  i, j[3];
PetscReal v[3];

MatCreate(PETSC_COMM_WORLD,&A);
MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4);
MatSetFromOptions(A);
MatSetUp(A);

i = 0;
j[0] = 0;    j[1] = 1;
v[0] = 4.0;  v[1] = -1.0;
MatSetValues(A,1,&i,2,j,v,INSERT_VALUES);
i = 1;
j[0] = 0;    j[1] = 1;    j[2] = 2;
v[0] = -1.0; v[1] = 4.0;  v[2] = -1.0;
MatSetValues(A,1,&i,3,j,v,INSERT_VALUES);
i = 2;
j[0] = 1;    j[1] = 2;    j[2] = 3;
MatSetValues(A,1,&i,3,j,v,INSERT_VALUES);
i = 3;
j[0] = 2;    j[1] = 3;
MatSetValues(A,1,&i,2,j,v,INSERT_VALUES);

MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  ierr = MatDestroy(&A); CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
//END
