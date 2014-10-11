
static char help[] = "Solves a structured-grid Poisson problem with a KSP.\n\
Input parameters include:\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";
// this is an edited form of src/ksp/ksp/examples/tutorials/ex2.c, but without
//   logging, some comments

/* performance:
$ for NN in 5 10 20 40 80 160; do ./c3poisson -m $NN -n $NN -ksp_rtol 1.0e-14; done
m=5, n=5
relative norm of error 1.18643e-15 iterations 12
m=10, n=10
relative norm of error 2.36879e-15 iterations 18
m=20, n=20
relative norm of error 3.36493e-15 iterations 29
m=40, n=40
relative norm of error 6.16418e-14 iterations 68
m=80, n=80
relative norm of error 1.84437e-13 iterations 203
m=160, n=160
relative norm of error 6.57092e-13 iterations 465

$ for NN in 5 10 20 40 80 160; do ./c3poisson -m $NN -n $NN -ksp_rtol 1.0e-14 -ksp_type cg; done
m=5, n=5
relative norm of error 1.22044e-15 iterations 12
m=10, n=10
relative norm of error 2.42097e-15 iterations 18
m=20, n=20
relative norm of error 3.49911e-15 iterations 29
m=40, n=40
relative norm of error 2.92444e-15 iterations 52
m=80, n=80
relative norm of error 8.49655e-15 iterations 97
m=160, n=160
relative norm of error 1.81329e-14 iterations 185
*/

#include <petscksp.h>
#include "convenience.h"

int main(int argc,char **args)
{
  Vec            b,u,uexact; // RHS, approx solution, exact solution
  Mat            A;          // linear system matrix
  KSP            ksp;        // linear solver context
  PetscInt       i,j,Ii,J,Istart,Iend,its;
  PetscScalar    norm, normexact, v;

  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

// FIXME: preferred, but default bug
#if 0
  PetscInt m,n;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for c3poisson", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-m","number of points in x direction", "", 5, &m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","number of points in y direction", "", 5, &n, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
#endif

  // FIXME: not preferred
  PetscInt m = 5, n = 5;
  ierr = PetscOptionsGetInt("", "-m", &m, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt("", "-n", &n, NULL); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"m=%d, n=%d\n",m,n); CHKERRQ(ierr);

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
  ierr = VecNorm(uexact,NORM_2,&normexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_2,&norm); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"relative norm of error %g iterations %D\n",
               norm/normexact,its); CHKERRQ(ierr);

  // free work space and finalize
  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&A);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
