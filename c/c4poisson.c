
static char help[] = "Solves a structured-grid Poisson problem with DMDA and KSP,\n"
"but (unlike c2poisson) also using KSPSetComputeOperators() so we can use\n"
"multigrid preconditioning at the command line.\n\n";

// based on src/ksp/ksp/examples/tutorials/ex50.c, but in Dirichlet-only case,
//   and following c2poisson.c closely

// SHOWS c2 AND c4 CODES ARE DOING SAME THING
//   $ ./c2poisson 
//   on 10 x 10 grid:  error |u-uexact|_inf = 0.000621778
//   $ ./c4poisson 
//   on 10 x 10 grid:  error |u-uexact|_inf = 0.000621778

// REGARDLESS OF INTEGER DIMS OF INITIAL GRID, THIS ALWAYS WORKS:
//   ./c4poisson -pc_type mg -pc_mg_levels N -da_refine N
// SO CONVERGENCE IS:
//   for NN in 1 2 3 4 5 6 7 8; do ./c4poisson -da_grid_x 5 -da_grid_y 5 -da_refine $NN -pc_type mg -pc_mg_levels $NN -ksp_rtol 1.0e-8; done

// USE MULTIGRID AND SHOW IT GRAPHICALLY:
//   ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause 1

// PERFORMANCE ANALYSIS; COMPARE c2poisson VERSION:
//   export PETSC_ARCH=linux-gnu-opt
//   make c4poisson
//   ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 9 -log_summary|grep "Solve: "
//   mpiexec -n 6 FIXME

// NOT SURE WHAT IS BEING ACCOMPLISHED HERE:
//   ./c4poisson -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
//   ./c4poisson -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor

// I NEED TO KNOW/UNDERSTAND EVERYTHING IN foo.txt:
//   mpiexec -n 4 ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -dm_view -ksp_view -log_summary &> foo.txt

// ONLY 17 SECONDS BUT USES 9 GB MEMORY:
//   mpiexec -n 4 ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 11 -ksp_monitor
// (COMPARE: mpiexec -n 4 ./c2poisson -da_grid_x 4097 -da_grid_y 4097 -ksp_type cg
// WHICH DOES 3329 ITERATIONS)

// COMPARABLE?:
//mpiexec -n 4 ./c4poisson -da_grid_x 4097 -da_grid_y 4097 -pc_type mg -pc_mg_levels 11 -ksp_monitor -pc_mg_type full -ksp_rtol 1.0e-12 -mg_levels_ksp_type cg

#include <math.h>
#include <petscdmda.h>
#include <petscksp.h>
#include "structuredlaplacian.h"

//COMPUTES
PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx) {
  PetscErrorCode ierr;
  DM             da;
  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = formRHS(da,b); CHKERRQ(ierr);
  return 0;
}


PetscErrorCode ComputeA(KSP ksp,Mat J, Mat A,void *ctx) {
  PetscErrorCode ierr;
  DM             da;
  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da); CHKERRQ(ierr);
  ierr = formdirichletlaplacian(da,1.0,A); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//ENDCOMPUTES


//MAIN
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,(char*)0,help);

  DM da;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                -10,-10,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

  // create linear solver context; compare to c2poisson.c version; note there
  // is no "Assemble" stage!; that happens inside KSPSolve()
  KSP ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,(DM)da);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeA,NULL);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  Vec u;
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  PetscLogStage  stage; //STRIP
  ierr = PetscLogStageRegister("Solve", &stage); CHKERRQ(ierr); //STRIP
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr); //STRIP
  ierr = KSPSolve(ksp,NULL,u);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr); //STRIP

  PetscScalar    errnorm;
  DMDALocalInfo  info;
  Vec            uexact;
  ierr = DMCreateGlobalVector(da,&uexact);CHKERRQ(ierr);
  ierr = formExact(da,uexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d grid:  error |u-uexact|_inf = %g\n",
             info.mx,info.my,errnorm); CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
//ENDMAIN

