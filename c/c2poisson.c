
static char help[] = "Solves a structured-grid Poisson problem with DMDA and KSP.\n\n";
// this is an edited form of src/ksp/ksp/examples/tutorials/ex46.c; see also ex2.c

// SHOW MAT:  ./c3poisson -a_mat_view
// SHOW MAT GRAPHICAL:  ./c3poisson -a_mat_view draw -draw_pause 5
// SHOW MAT DENSE:  ./c3poisson -da_grid_x 3 -da_grid_y 3 -a_mat_view ::ascii_dense

// FIXME: this convergence is for discrete linear solver only, not for PDE
// CONVERGENCE: for NN in 5 10 20 40 80 160; do ./c3poisson -da_grid_x $NN -da_grid_y $NN -ksp_rtol 1.0e-14 -ksp_type cg; done

// VISUALIZATION OF SOLUTION: mpiexec -n 6 ./c3poisson -ksp_rtol 1.0e-12 -da_grid_x 129 -da_grid_y 129 -ksp_type cg -ksp_monitor_solution

// PURE LU ALGORITHM: ./c3poisson  -ksp_type preonly -pc_type lu
// PURE CHOLESKY ALGORITHM: ./c3poisson  -ksp_type preonly -pc_type cholesky
// PURE CG ALGORITHM:
//   ./c3poisson  -ksp_type cg -pc_type none -ksp_view  # JUST SHOW KSP STRUCTURE
//   ./c3poisson -da_grid_x 257 -da_grid_y 257 -ksp_type cg -pc_type none -log_summary
//   (compare Elman p.72 and Algorithm 2.1 = cg: "The computational work of one
//   iteration is two inner products, three vector updates, and one matrix-vector
//   product.")

// PERFORMANCE ANALYSIS:
//   export PETSC_ARCH=linux-gnu-opt
//   make c3poisson
//   ./c3poisson -da_grid_x 513 -da_grid_y 513 -ksp_type cg -log_summary
//   mpiexec -n 4 ./c3poisson -da_grid_x 513 -da_grid_y 513 -ksp_type cg -log_summary

#include <petscdmda.h>
#include <petscksp.h>
#include "convenience.h"

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

//CREATEGRID
  // create distributed array to handle parallel distribution.
  // default size (5 x 5) can be changed using -da_grid_x M -da_grid_y N
  DM             da;
  DMDALocalInfo  info;
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
//ENDCREATEGRID

//CREATEMATRIX
  // create linear system matrix
  // to use symmetric storage, run with -dm_mat_type sbaij -mat_ignore_lower_triangular ??
  Mat  A;
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);

  // assemble matrix based on local stencil
  PetscInt       i, j;
  PetscScalar    hx = 1./info.mx,  hy = 1./info.my;  // domain is [0,1]x[0,1]
  PetscLogStage  stage;
  ierr = PetscLogStageRegister("Assembly", &stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscScalar v[5];
      PetscInt    ncols = 0;
      row.j        = j; row.i = i;
      col[ncols].j = j; col[ncols].i = i; v[ncols++] = 2*(hx/hy + hy/hx);
      if (i>0)         {col[ncols].j = j;   col[ncols].i = i-1; v[ncols++] = -hy/hx;}
      if (i<info.mx-1) {col[ncols].j = j;   col[ncols].i = i+1; v[ncols++] = -hy/hx;}
      if (j>0)         {col[ncols].j = j-1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      if (j<info.my-1) {col[ncols].j = j+1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  matassembly(A)
  ierr = PetscLogStagePop();CHKERRQ(ierr);
//ENDCREATEMATRIX

//SOLVE
  // create RHS, approx solution, exact solution
  Vec  b,u,uexact;
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = VecSet(uexact,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uexact,b); CHKERRQ(ierr);

  // create linear solver context
  KSP  ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // solve
  ierr = PetscLogStageRegister("Solve", &stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  // report on ksp iterations and measure DISCRETE error in solution
  PetscInt     its;
  PetscScalar  norm, normexact;
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = VecNorm(uexact,NORM_2,&normexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_2,&norm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d grid:  iterations %D, error |u-uexact|_2/|uexact|_2 = %g\n",
             info.mx,info.my,its,norm/normexact); CHKERRQ(ierr);
//ENDSOLVE

  // free work space and finalize
  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);
  MatDestroy(&A);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
