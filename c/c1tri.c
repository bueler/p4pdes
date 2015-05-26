
/* shows:
       PetscOptionsXXX()
       XXXSetOptionsPrefix()
       VecDuplicate()
       MatXXXAIJSetPreallocation() or MatSetUp()
       GetOwnershipRange()
       MatMult()   for exact solution
       VecAXPY() and VecNorm()  for computing error
*/

static char help[] = "Solve a symmetric tridiagonal system of arbitrary size.\n"
                     "Option prefix = tri_.\n";

#include <petsc.h>

int main(int argc,char **args) {
  Vec            x, b, xexact;
  Mat            A;
  KSP            ksp;
  PetscBool      prealloc = PETSC_FALSE;
  PetscInt       N = 4, i, Istart, Iend, j[3];
  PetscReal      v[3], xval, err;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"tri_","options for c1tri",""); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-prealloc",
                          "use MatMPIAIJSetPreallocation() instead of MatSetUp()",
                          NULL,prealloc,&prealloc,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n",
                         "dimension of linear system",NULL,N,&N,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N); CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(x,"x_"); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xexact); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"A"); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);

  if (prealloc) {
    int size;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSeqAIJSetPreallocation(A,3,NULL); CHKERRQ(ierr);
    } else {
      ierr = MatMPIAIJSetPreallocation(A,3,NULL,1,NULL); CHKERRQ(ierr);
    }
  } else {
    ierr = MatSetUp(A); CHKERRQ(ierr);
  }

  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    if (i == 0) {
      v[0] = 3.0;  v[1] = -1.0;
      j[0] = 0;    j[1] = 1;
      ierr = MatSetValues(A,1,&i,2,j,v,INSERT_VALUES); CHKERRQ(ierr);
    } else {
      v[0] = -1.0;  v[1] = 3.0;  v[2] = -1.0;
      j[0] = i-1;   j[1] = i;    j[2] = i+1;
      if (i == N-1) {
        ierr = MatSetValues(A,1,&i,2,j,v,INSERT_VALUES); CHKERRQ(ierr);
      } else {
        ierr = MatSetValues(A,1,&i,3,j,v,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
    xval = exp(cos(i));
    ierr = VecSetValues(xexact,1,&i,&xval,INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(xexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xexact); CHKERRQ(ierr);

  ierr = MatMult(A,xexact,b); CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);

  ierr = VecAXPY(x,-1.0,xexact); CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"error |x-xexact|_2 = %.1e\n",err); CHKERRQ(ierr);

  KSPDestroy(&ksp);
  VecDestroy(&x);  VecDestroy(&b);  VecDestroy(&xexact);
  MatDestroy(&A);
  PetscFinalize();
  return 0;
}

