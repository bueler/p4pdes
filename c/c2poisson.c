
static char help[] = "Solves a structured-grid Poisson problem with DMDA and KSP.\n\n";

// SEE ALSO:  src/ksp/ksp/examples/tutorials/ex50.c
// IT IS SIMILAR BUT HAS MULTIGRID ABILITY BECAUSE OPERATOR A IS GENERATED AT
// EACH LEVEL THROUGH
//     KSPSetDM(ksp,(DM)da) ... KSPSetComputeOperators(ksp,ComputeJacobian,&user)
// AND ComputeJacobian() USER CODE

// SHOW MAT DENSE:  ./c2poisson -da_grid_x 3 -da_grid_y 3 -a_mat_view ::ascii_dense
// SHOW MAT GRAPHICAL:  ./c2poisson -a_mat_view draw -draw_pause 5

// CONVERGENCE:
//   for NN in 5 9 17 33 65 129 257; do ./c2poisson -da_grid_x $NN -da_grid_y $NN -ksp_rtol 1.0e-8 -ksp_type cg; done

// SAME CONVERGENCE USING -da_refine:
//   for NN in 1 2 3 4 5 6; do ./c2poisson -da_grid_x 5 -da_grid_y 5 -ksp_rtol 1.0e-8 -ksp_type cg -da_refine $NN; done

// VISUALIZATION OF SOLUTION: mpiexec -n 6 ./c2poisson -da_grid_x 129 -da_grid_y 129 -ksp_type cg -ksp_monitor_solution

// PERFORMANCE ON SAME:
//   for NN in 5 9 17 33 65 129 257; do ./c2poisson -da_grid_x $NN -da_grid_y $NN -ksp_rtol 1.0e-8 -ksp_type cg -log_summary|grep "Time (sec):"; done

// WEAK SCALING IN TERMS OF FLOPS ONLY:
//   for kk in 0 1 2 3; do NN=$((50*(2**$kk))); MM=$((2**(2*$kk))); cmd="mpiexec -n $MM ./c2poisson -da_grid_x $NN -da_grid_y $NN -ksp_rtol 1.0e-8 -ksp_type cg -log_summary"; echo $cmd; $cmd |'grep' "Flops:  "; echo; done

// SHOW KSP STRUCTURE:  ./c2poisson -ksp_view

// DIRECT LINEAR SOLVERS:
//   LU ALGORITHM: ./c2poisson -ksp_type preonly -pc_type lu
//   CHOLESKY ALGORITHM: ./c2poisson -ksp_type preonly -pc_type cholesky

// UNPRECONDITIONED CG ALGORITHM:
//   ./c2poisson -ksp_type cg -pc_type none -ksp_view  # JUST SHOW KSP STRUCTURE
//   ./c2poisson -da_grid_x 257 -da_grid_y 257 -ksp_type cg -pc_type none -log_summary
//   (compare Elman p.72 and Algorithm 2.1 = cg: "The computational work of one
//   iteration is two inner products, three vector updates, and one matrix-vector
//   product.")

// PERFORMANCE ANALYSIS:
//   export PETSC_ARCH=linux-gnu-opt
//   make c2poisson
//   ./c2poisson -da_grid_x 1029 -da_grid_y 1029 -ksp_type cg -log_summary|grep "Solve: "
//   mpiexec -n 6 ./c2poisson -da_grid_x 1029 -da_grid_y 1029 -ksp_type cg -log_summary|grep "Solve: "


#include <math.h>
#include <petscdmda.h>
#include <petscksp.h>
#include "structuredlaplacian.h"

//RHS
PetscErrorCode formRHSandExact(DM da, Vec b, Vec uexact) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  PetscInt       i, j;
  PetscReal      hx = 1./(double)(info.mx-1),
                 hy = 1./(double)(info.my-1),  // domain is [0,1] x [0,1]
                 pi = PETSC_PI, x, y, f, **ab, **auex;
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, uexact, &auex);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * hy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * hx;
      // choose exact solution to satisfy boundary conditions, and be a bit
      //   generic (e.g. not equal to an eigenvector)
      auex[j][i] = x * (1.0 - x) * sin(3.0 * pi * y);
      if ( (i>0) && (i<info.mx-1) && (j>0) && (j<info.my-1) ) { // if not bdry
        // f = - (u_xx + u_yy)
        f = 2 * sin(3.0 * pi * y) + 9.0 * pi * pi * auex[j][i];
        ab[j][i] = hx * hy * f;
      } else {
        ab[j][i] = 0.0;                          // on bdry we have "1 * u = 0"
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &ab);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, uexact, &auex);CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  return 0;
}
//ENDRHS


int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

//CREATE
  // default size (10 x 10) can be changed using -da_grid_x M -da_grid_y N
  DM             da;
  PetscLogStage  stage;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,  // points on boundary have no
                DMDA_STENCIL_STAR,-10,-10,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

  // create linear system matrix
  // to use symmetric storage, run with -dm_mat_type sbaij -mat_ignore_lower_triangular ??
  Mat  A;
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Matrix Assembly", &stage); CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr);
  ierr = formdirichletlaplacian(da,1.0,A); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);
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

  // report on grid, ksp results, and numerical error
  PetscInt       its;
  PetscScalar    resnorm, errnorm;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its); CHKERRQ(ierr);
  ierr = KSPGetResidualNorm(ksp,&resnorm); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %4d x %4d grid:  iterations %D, residual norm = %g,\n"
             "                      error |u-uexact|_inf = %g\n",
             info.mx,info.my,its,resnorm,errnorm); CHKERRQ(ierr);
//ENDSOLVE

  KSPDestroy(&ksp);  MatDestroy(&A);
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  PetscFinalize();
  return 0;
}

