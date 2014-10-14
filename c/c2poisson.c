
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
#include "structuredlaplacian.h"

//RHS
PetscErrorCode formRHSandExact(DM da, Vec b, Vec uexact) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  PetscInt       i, j;
  PetscReal      hx = 1./info.mx,  hy = 1./info.my,  // domain is [0,1] x [0,1]
                 x, y, **ab, **auex;
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, uexact, &auex);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * hy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * hx;
      auex[j][i] =
      ab[j][i]   = 
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &ab);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, uexact, &auex);CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(b,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
//ENDRHS


int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

//CREATE
  // default size (5 x 5) can be changed using -da_grid_x M -da_grid_y N
  DM             da;
  PetscLogStage  stage;
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);

  // create linear system matrix
  // to use symmetric storage, run with -dm_mat_type sbaij -mat_ignore_lower_triangular ??
  Mat  A;
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Matrix Assembly", &stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr);
  ierr = formlaplacian(da,A); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
//ENDCREATE

//SOLVE
  // create right-hand-side (RHS), approx solution, exact solution; fill
  Vec  b,u,uexact;
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = formRHSandExact(da,b,uexact); CHKERRQ(ierr);

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

  // report on grid, ksp iterations, and numerical error
  PetscInt       its;
  PetscScalar    norm, normexact;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = VecNorm(uexact,NORM_2,&normexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_2,&norm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d grid:  iterations %D, error |u-uexact|_2/|uexact|_2 = %g\n",
             info.mx,info.my,its,norm/normexact); CHKERRQ(ierr);

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}
//ENDSOLVE
