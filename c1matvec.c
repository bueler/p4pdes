static char help[] = "Solves a symmetric 4x4 linear system with KSP.\n\
Try:\n\
  ./c1matvec                            # sequential solve on one process\n\
  mpiexec -n 2 ./c1matvec               # parallel solve on two processes\n\
  ./c1matvec -ksp_monitor               # show residual norms\n\
  ./c1matvec -mat_view                  # show the sparse storage form\n\
  ./c1matvec -ksp_view                  # show default KSP type, etc.\n\
  ./c1matvec -ksp_monitor -ksp_type cg  # try CG instead of default GMRES\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,xexact;
  Mat            A;
  KSP            ksp;

  PetscInitialize(&argc,&args,(char*)0,help);

  MatCreate(PETSC_COMM_WORLD,&A);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4);
  MatSetFromOptions(A);
  MatSetUp(A);

  PetscInt       i,j,Istart,Iend;
  PetscScalar    v;
  MatGetOwnershipRange(A,&Istart,&Iend);
  for (i=Istart; i<Iend; i++) {
    v = -2.0;  j = i;  MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
    if (i > 0) {
      v = 1.0;  j = i-1;  MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
    }
    if (i < 3) {
      v = 1.0;  j = i+1;  MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);
    }
  }
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);

  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x,PETSC_DECIDE,4);
  VecSetFromOptions(x);
  PetscObjectSetName((PetscObject)x,"approximate solution");
  VecDuplicate(x,&b);
  VecDuplicate(x,&xexact);
  PetscObjectSetName((PetscObject)xexact,"exact solution");

  PetscInt       ix[4] = {0.0, 1.0, 2.0, 3.0};
  PetscScalar    xexactvals[4] = {3.0, 2.0, 1.0, 0.0};
  VecSetValues(xexact,4,ix,xexactvals,INSERT_VALUES);
  VecAssemblyBegin(xexact);  VecAssemblyEnd(xexact);
  MatMult(A,xexact,b);

  KSPCreate(PETSC_COMM_WORLD,&ksp);
  KSPSetOperators(ksp,A,A);
  KSPSetFromOptions(ksp);

  KSPSolve(ksp,b,x);

  VecView(x,PETSC_VIEWER_STDOUT_WORLD);
  VecView(xexact,PETSC_VIEWER_STDOUT_WORLD);

  KSPDestroy(&ksp);  VecDestroy(&x);  VecDestroy(&b);  MatDestroy(&A);
  PetscFinalize();
  return 0;
}
