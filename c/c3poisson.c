
static char help[] = "Solves a structured-grid Poisson problem with a KSP.\n\
Input parameters include:\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";
// this is an edited form of src/ksp/ksp/examples/tutorials/ex2.c, but without
//   logging, some comments

/* 
$ for NN in 5 10 20 40 80 160; do ./c3poisson -m $NN -n $NN -ksp_rtol 1.0e-14 -ksp_type cg; done
*/

#include <petscksp.h>
#include "convenience.h"

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

  // read options which determine mesh
  PetscInt m = 5, n = 5; // because of design in PetscOptionsInt(), can't set defaultv=5
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for c3poisson", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m","number of points in x direction", "", m, &m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","number of points in y direction", "", n, &n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // create linear system matrix
  Mat  A;
  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL); CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  // assemble matrix, by rows owned by process
  PetscInt    i,j,Ii,J,Istart,Iend;
  PetscScalar v;
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

  // create RHS, approx solution, exact solution
  Vec  b,u,uexact;
  ierr = VecCreate(PETSC_COMM_WORLD,&b); CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,m*n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(b); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = VecSet(uexact,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uexact,b); CHKERRQ(ierr);

  // create linear solver context and solve
  KSP  ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-2/((m+1)*(n+1)),1.e-50,PETSC_DEFAULT,
                          PETSC_DEFAULT); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);

  // report on ksp iterations and measure DISCRETE error in solution
  PetscInt     its;
  PetscScalar  norm, normexact;
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = VecNorm(uexact,NORM_2,&normexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_2,&norm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d grid:  iterations %D, error |u-uexact|_2/|uexact|_2 = %g\n",
             m,n,its,norm/normexact); CHKERRQ(ierr);

  // free work space and finalize
  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&A);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
