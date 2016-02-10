
static char help[] =
"Scalar ODE solver by TS.  Serial only.  Option prefix -ode_.\n"
"Solves dy/dt = f(t,y) with y(t0) = y0 to compute y(tf), where t0, y0, tf are\n"
"all set by options.  Implemented example has f(t,y) = 2 t, so y(1) = 1.\n\n";

#include <petsc.h>

PetscErrorCode FormRHSFunction(TS ts, double t, Vec y, Vec f, void *ptr) {
  const double *ay;
  double       *af;
  VecGetArrayRead(y,&ay);
  VecGetArray(f,&af);
  af[0] = 2.0 * t;     // could be any f(t,ay[0])
  VecRestoreArrayRead(y,&ay);
  VecRestoreArray(f,&af);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  double    t0 = 0.0,  tf = 1.0,  y0 = 0.0,  dtinitial;
  int       steps = 10;
  PetscBool interpolate = PETSC_FALSE;
  Vec       y;
  TS        ts;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ode_", 
                           "options for scalar ODE solver ode.c", ""); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate","ask TS to interpolate at final time","ode.c",
                          interpolate,&interpolate,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps","desired number of time-steps","ode.c",
                         steps,&steps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","initial time","ode.c",t0,&t0,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","final time","ode.c",tf,&tf,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-y0","initial value of y(t)","ode.c",y0,&y0,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y); CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,1); CHKERRQ(ierr);
  ierr = VecSetFromOptions(y); CHKERRQ(ierr);
  ierr = VecSet(y,y0); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);

  ierr = TSSetType(ts,TSEULER); CHKERRQ(ierr);
  //dtinitial = 1.0123456789*(tf-t0)/(double)steps;    // magic number to fix issue #119
  dtinitial = (tf-t0)/(double)steps;
  ierr = TSSetDuration(ts,100*steps,tf-t0); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,
    (interpolate) ? TS_EXACTFINALTIME_INTERPOLATE: TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,t0,dtinitial); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "solving from t0 = %g to tf = %g with y0 = %g and initial dt = %g ...\n",
             t0,tf,y0,dtinitial); CHKERRQ(ierr);
  ierr = TSSolve(ts,y); CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

