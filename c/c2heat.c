
static char help[] = "Time-dependent heat equation in 2d.\n";

#include <petscdmda.h>
#include <petscts.h>
#include "structuredlaplacian.h"

extern PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode FormSolutionAtTime(DM, PetscReal, Vec);
extern PetscErrorCode MyTSMonitor(TS, PetscInt, PetscReal, Vec, void*);

int main(int argc,char **argv)
{
  TS             ts;                  // time integrator
  DM             da;                  // grid topology
  Vec            u;                   // solution vector
  PetscInt       maxsteps = 10;
  PetscReal      t0 = 0.0, tf = 0.01;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,-5,-5,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);

  Mat  A;
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = formlaplacian(da,A); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);

  ierr = TSMonitorSet(ts,MyTSMonitor,0,0);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);

  ierr = FormSolutionAtTime(da,t0,u);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,t0,(tf-t0)/(double)maxsteps);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,tf);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,u);CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}


// since ODE sys is  du/dt = A u,  just compute "G(t,u) = A u"
// FIXME: would like more generic  G(t,u) = a(t,x) A u + f(t,x) ??
PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec G, void *ptr) {
  PetscErrorCode ierr;
  Mat            A;
  PetscReal      hx,hy;
  DM             da;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(info.mx-1);
  hy = 1.0/(PetscReal)(info.my-1);
  ierr = TSGetRHSJacobian(ts,&A,NULL,NULL,NULL); CHKERRQ(ierr);
  ierr = MatMult(A,X,G); CHKERRQ(ierr);
  ierr = VecScale(G,-1.0/(hx*hy)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// use both for initial condition at t=0 and exact solution at t=tf
PetscErrorCode FormSolutionAtTime(DM da, PetscReal t, Vec U) {
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      **u, pi = PETSC_PI, hx,hy,x,y,
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
      x = i*hx;
      u[j][i] =  c1 * PetscExpReal(-(K1*K1+L1*L1)*pi*pi * t) * sin(K1*pi*x) * sin(L1*pi*y);
      u[j][i] += c2 * PetscExpReal(-(K2*K2+L2*L2)*pi*pi * t) * sin(K2*pi*x) * sin(L2*pi*y);
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MyTSMonitor(TS ts,PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal      norm;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  ierr = VecNorm(v,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ts,&comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"t = %8g:   max |u(t,x,y)| = %g\n",ptime,norm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

