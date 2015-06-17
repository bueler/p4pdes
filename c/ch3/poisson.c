
static char help[] = "Solves a structured-grid Poisson problem with DMDA and KSP.\n\n";

// PRECONDITIONED CG:  look at iterations in IC(0) preconditioned,

//   for NN in 1 2 3 4 5; do ./poisson -da_refine $NN -ksp_rtol 1.0e-8 -ksp_type cg -pc_type none -ksp_converged_reason; done
// ITERATIONS (ASYMPTOTICALLY) DOUBLE WITH EACH GRID REFINEMENT
//   (compare Elman p. 76: "...suggests that for uniformly refined grids, the
//   number of CG iterations required to meet a fixed tolerance will approximately
//   double with each grid refinement"

//   for NN in 1 2 3 4 5; do ./poisson -da_refine $NN -ksp_rtol 1.0e-8 -ksp_type cg -pc_type icc -ksp_converged_reason; done
// compare Elman p. 82: "One known result [ABOUT IC(0) PRECONDITIONING USED
//   IN SECOND CASE ABOVE] is that the asymptotic behavior of the condition number
//   using IC(0) preconditioning is unchanged: \kappa(M^{-1} A) = O(h^{-2})."
//   THIS IS WHAT I SEE!!)

// MINRES VS CG:
// for NN in 1 2 3 4 5; do ./poisson -da_refine $NN -ksp_rtol 1.0e-8 -ksp_type minres -pc_type none -ksp_converged_reason; done
//   (compare Elman p. 88: "Indeed, when solving discrete Poisson problems the
//   the convergence of MINRES is almost identical to that of CG"  THIS IS WHAT I SEE!!)

// CHOLESKY VS CG:
// for NN in 1 2 3 4 5; do timer ./poisson -da_refine $NN -ksp_type preonly -pc_type cholesky -ksp_converged_reason; done
// for NN in 1 2 3 4 5; do timer ./poisson -da_refine $NN -ksp_type preonly -pc_type cholesky -ksp_converged_reason; done

// PERFORMANCE ANALYSIS:
//   export PETSC_ARCH=linux-gnu-opt
//   make poisson
//   ./poisson -da_grid_x 1025 -da_grid_y 1025 -ksp_type cg -log_summary|grep "Solve: "
//   mpiexec -n 6 ./poisson -da_grid_x 1025 -da_grid_y 1025 -ksp_type cg -log_summary|grep "Solve: "

// WEAK SCALING IN TERMS OF FLOPS ONLY:
//   for kk in 0 1 2 3; do NN=$((50*(2**$kk))); MM=$((2**(2*$kk))); cmd="mpiexec -n $MM ./poisson -da_grid_x $NN -da_grid_y $NN -ksp_rtol 1.0e-8 -ksp_type cg -log_summary"; echo $cmd; $cmd |'grep' "Flops:  "; echo; done


//CREATE
#include <petsc.h>
#include "structuredpoisson.h"

int main(int argc,char **args) {
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

  // default size (9 x 9) can be changed using -da_grid_x M -da_grid_y N
  DM            da;
  DMDALocalInfo info;
  PetscReal     hx, hy;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(info.mx-1);
  hy = 1.0/(info.my-1);

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
  ierr = formExact(da,info,hx,hy,uexact); CHKERRQ(ierr);
  ierr = formRHS(da,info,hx,hy,b); CHKERRQ(ierr);

  // assemble linear system
  PetscLogStage  stage; //STRIP
  ierr = PetscLogStageRegister("Matrix Assembly", &stage); CHKERRQ(ierr); //STRIP
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr); //STRIP
  ierr = formdirichletlaplacian(da,info,hx,hy,1.0,A); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr); //STRIP
//ENDCREATE

//SOLVE
  // create linear solver context
  KSP  ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // solve
  ierr = PetscLogStageRegister("Solve", &stage); CHKERRQ(ierr); //STRIP
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr); //STRIP
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr); //STRIP

  // report on grid and numerical error
  PetscScalar    errnorm;
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
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

