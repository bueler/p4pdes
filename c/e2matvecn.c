
static char help[] =
"Solution to exercise 2 in chapter 1.\n\n";

// try:
//   ./e2matvecn -pc_type none -ksp_type gmres -ksp_monitor_singular_value -ksp_gmres_restart 1000 -N 10 | grep "max/min" | tail -n 1

#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,xexact;
  Mat            A;
  KSP            ksp;
  PetscErrorCode  ierr;
  PetscInt  N=4;
  PetscBool Nset;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for e2matvecn", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-N","size of matrix\n", "", N, &N, &Nset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"approx_solution"); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  ierr = VecDuplicate(x,&b); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&xexact); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)xexact,"exact_solution"); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)A,"A"); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
//ENDSETUP

  PetscInt       i,j,Istart,Iend;
  PetscScalar    v;
  ierr = MatSetUp(A); CHKERRQ(ierr);  // called instead of preallocation
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    v = (PetscScalar)N-i-1;
    ierr = VecSetValues(xexact,1,&i,&v,INSERT_VALUES); CHKERRQ(ierr);
    v = -2.0;  j = i;
    ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    if (i > 0) {
      v = 1.0;  j = i-1;
      ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
    if (i < N-1) {
      v = 1.0;  j = i+1;
      ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

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
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
//END
