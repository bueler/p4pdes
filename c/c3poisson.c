
static char help[] = "Solves a structured-grid Poisson problem with a KSP.\n\
Input parameters include:\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";
// this is an edited form of src/ksp/ksp/examples/tutorials/ex2.c, but without
//   logging, some comments

#include <petscksp.h>
#include "convenience.h"

int main(int argc,char **args)
{
  Vec            b,u,uexact; // RHS, approx solution, exact solution
  Mat            A;          // linear system matrix
  KSP            ksp;        // linear solver context
  PetscInt       i,j,Ii,J,Istart,Iend,m,n,its;
  PetscScalar    norm, v;

  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for c3poisson", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m","number of points in x direction", "", 5, &m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","number of points in y direction", "", 5, &n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

PetscPrintf(PETSC_COMM_WORLD,"m=%d, n=%d\n",m,n);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    // set matrix elements for the 2-D, five-point stencil in parallel
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii - n;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i<m-1) {
      J = Ii + n;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (j>0) {
      J = Ii - 1;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (j<n-1) {
      J = Ii + 1;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    v = 4.0;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES); CHKERRQ(ierr);
  }
  matassembly(A)

  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);

  ierr = VecSet(uexact,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uexact,b); CHKERRQ(ierr);

  // create linear solver context and solve
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,
                          PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);

  // check solution and clean up
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_2,&norm); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",
               norm,its); CHKERRQ(ierr);

  // free work space and finalize
  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&A);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
