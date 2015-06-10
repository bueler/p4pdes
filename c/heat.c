
static char help[] = "Time-dependent heat equation in 2d.\n";

/*
  Solves heat equation
    u_t = u_xx + u_yy
  on the unit square S = (0,1) x (0,1) with homogeneous Dirichlet
  with initial condition
    u(0,x,y) = sin(pi x) sin(pi y) + sin(3 pi x) sin(2 pi y)

  Exact solution is
    u(t,x,y) = e^{- pi^2 t} sin(pi x) sin(pi y)
                 + e^{- (3^2 + 2^2) pi^2 t} sin(3 pi x) sin(2 pi y)

  Uses finite difference Laplacian.  Compare c2poisson.c

For examples, try

  ./c2heat
  ./c2heat -ts_type euler                           # same as above
  ./c2heat -steps 100                               #FIXME: disparity between tf sought and tf returned
  ./c2heat -ts_monitor                              # report on steps
  ./c2heat -da_grid_x 20 -da_grid_y 20              # higher res
  ./c2heat -da_grid_x 100 -da_grid_y 100            # higher; Euler failure
  ./c2heat -da_grid_x 100 -da_grid_y 100 -steps 500 # bring dt below stability limit
  ./c2heat -ts_type beuler                          #FIXME: see 10/18/2014 bug report
  ./c2heat -ts_monitor_draw_solution -draw_pause 1

*/

#include <petscdmda.h>
#include <petscts.h>
#include "ch3/structuredpoisson.h"


// use both for initial condition at t=0 and exact solution at t=tf
PetscErrorCode FormSolutionAtTime(DM da, PetscReal t, Vec U) {
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      **u,hx,hy,x,y,
                 pi = PETSC_PI,
                 c1 = 1.0, K1 = 1.0, L1 = 1.0,
                 c2 = 1.0, K2 = 3.0, L2 = 2.0;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);
  hy = 1.0/(PetscReal)(info.my-1);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j*hy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      if ( (i == 0) || (i == info.mx-1) || (j == 0) || (j == info.my-1) )
        u[j][i] = 0.0;
      else {
        x = i*hx;
        u[j][i] =  c1 * PetscExpReal(-(K1*K1+L1*L1)*pi*pi * t) * sin(K1*pi*x) * sin(L1*pi*y);
        u[j][i] += c2 * PetscExpReal(-(K2*K2+L2*L2)*pi*pi * t) * sin(K2*pi*x) * sin(L2*pi*y);
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc,char **argv) {
  TS             ts;                  // time integrator
  DM             da;                  // grid topology
  Vec            u, uexact;           // numerical and exact solution vectors
  PetscInt       steps = 10;
  PetscReal      t0 = 0.0, tf = 0.01, errnorm;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for c2heat", ""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps",
                         "choose number of requested time steps\n", "", steps,
                         &steps, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // CREATE GRID
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);

  // CREATE A, CHANGING SCALING RELATIVE TO POISSON PROBLEM
  Mat  A;
  PetscReal      hx,hy,dtdefault;
  DMDALocalInfo  info;
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);
  hy = 1.0/(PetscReal)(info.my-1);

  ierr = formdirichletlaplacian(da,info,hx,hy,0.0,A); CHKERRQ(ierr);
  ierr = MatScale(A,-1.0/(hx*hy)); CHKERRQ(ierr);

  // CREATE AND SETUP TS
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);

  // SET INITIAL CONDITION AND AS FOR DURATION
  ierr = FormSolutionAtTime(da,t0,u);CHKERRQ(ierr);
  dtdefault = (tf-t0) / (double)steps;
  ierr = TSSetInitialTimeStep(ts,t0,dtdefault);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10*steps,tf);CHKERRQ(ierr);
  //FIXME: seems to have no effect
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %4d x %4d grid, from t0 = %g to tf = %g, with dt0 = %f:\n",
             info.mx,info.my,t0,tf,dtdefault); CHKERRQ(ierr);
  //FIXME: remove this when -ts_type beuler works
  PetscReal dtEuler;
  dtEuler = 1.0/(hx*hx) + 1.0/(hy*hy);
  dtEuler = 1.0 / (2.0 * dtEuler);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "stability limit for Euler:  dt = %f:\n",
             dtEuler); CHKERRQ(ierr);

  // SOLVE (USING USER'S OPTIONS)
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,u);CHKERRQ(ierr);

  // show result:
  // FIXME: does not return tf, as promised in man page for TSGetSolveTime()
  PetscReal tfreturned;
  ierr = TSGetSolveTime(ts,&tfreturned);CHKERRQ(ierr);

  ierr = FormSolutionAtTime(da,tfreturned,uexact);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "result at t = %g (note |tf-tr_returned| = %e):     error |u-uexact|_inf = %g\n",
             tfreturned,fabs(tf-tfreturned),errnorm); CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

