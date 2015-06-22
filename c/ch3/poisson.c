
//CREATE
static char help[] = "Solves a structured-grid Poisson problem with DMDA and KSP.\n\n";

#include <petsc.h>
#include "structuredpoisson.h"

int main(int argc,char **args) {
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

  // default size (9 x 9) can be changed using -da_grid_x M -da_grid_y N
  DM  da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

  // create linear system matrix A
  Mat  A;
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);

  // create right-hand-side (RHS) b, approx solution u, exact solution uexact
  Vec  b,u,uexact;
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);

  // fill known vectors
  ierr = formExact(da,uexact); CHKERRQ(ierr);
  ierr = formRHS(da,b); CHKERRQ(ierr);

  // assemble linear system
  ierr = formdirichletlaplacian(da,1.0,A); CHKERRQ(ierr);
//ENDCREATE

//SOLVE
  // create linear solver context
  KSP  ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // solve
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);

  // report on grid and numerical error
  PetscScalar    errnorm;
  DMDALocalInfo  info;
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d grid:  error |u-uexact|_inf = %g\n",
             info.mx,info.my,errnorm); CHKERRQ(ierr);

  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  MatDestroy(&A);
  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDSOLVE

