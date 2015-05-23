static char help[] = "Solve a symmetric tridiagonal system. User chooses size.\n";

FIXME: does not compile, and much coding needed

FIXME
shows: PetscObjectSetName, MatSetOptionsPrefix, options,
       VecDuplicate, SetPreallocation, GetOwnershipRange, MatMult,
       VecNorm (for error)
       Destroy methods do not need error handling

#include <petsc.h>

int main(int argc,char **args) {
  Vec            x,b,xexact;
  Mat            A;
  KSP            ksp;
  PetscInt       N,i,j,Istart,Iend;
  PetscReal      v;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,4); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"approx_solution"); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xexact); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)xexact,"exact_solution"); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"A"); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
//ENDSETUP

  ierr = MatSetUp(A); CHKERRQ(ierr);  // called instead of preallocation
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    v = -2.0;  j = i;
    ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    if (i > 0) {
      v = 1.0;  j = i-1;
      ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i < 3) {
      v = 1.0;  j = i+1;
      ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

  PetscInt       ix[4] = {0, 1, 2, 3};
  PetscScalar    xexactvals[4] = {3.0, 2.0, 1.0, 0.0};
  ierr = VecSetValues(xexact,4,ix,xexactvals,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(xexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xexact); CHKERRQ(ierr);
  ierr = MatMult(A,xexact,b); CHKERRQ(ierr);
//ENDASSEMBLY

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  ierr = KSPSolve(ksp,b,x); CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecView(xexact,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = VecDestroy(&xexact); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
//END
