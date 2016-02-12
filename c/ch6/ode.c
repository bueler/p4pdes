static char help[] =
"Scalar ODE solver by TS.  Option prefix -ode_.\n"
"Solves dy/dt = g(t,y) with y(t0) = y0 to compute y(tf), where t0, y0, tf are\n"
"all set by options.  Serial only.\n"
"Implemented example has g(t,y) = t + y, so y(1) = e-2 = 0.7182818.\n\n";

#include <petsc.h>

PetscErrorCode FormRHSFunction(TS ts, double t, Vec y, Vec g, void *ptr) {
    const double *ay;
    double       *ag;
    VecGetArrayRead(y,&ay);
    VecGetArray(g,&ag);
    ag[0] = t + ay[0];            // could be any g(t,ay[0])
    VecRestoreArrayRead(y,&ay);
    VecRestoreArray(g,&ag);
    PetscFunctionReturn(0);
}

PetscErrorCode FormRHSJacobian(TS ts, double t, Vec y, Mat J, Mat P, void *ptr) {
    PetscErrorCode ierr;
    const double *ay;
    double       v;
    int          row = 0, col = 0;

    ierr = VecGetArrayRead(y,&ay); CHKERRQ(ierr);
    v = 1.0;                     // entry of dg/dy, a 1x1 matrix
    ierr = MatSetValues(P,1,&row,1,&col,&v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(y,&ay); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  double    t0 = 0.0,  tf = 1.0,  y0 = 0.0,  dtinitial;
  int       steps = 10;
  Vec       y;
  Mat       J;
  TS        ts;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ode_", 
                           "options for scalar ODE solver ode.c", ""); CHKERRQ(ierr);
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

  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,1,1); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL); CHKERRQ(ierr);

  // dtinitial = 1.000000123456789*(tf-t0)/(double)steps; //STRIP magic number fix for PETSc issue #119 for Euler
  dtinitial = (tf-t0)/(double)steps;
  ierr = TSSetDuration(ts,100*steps,tf-t0); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,t0,dtinitial); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "solving t0 = %.3f to tf = %.3f with y0 = %.3f and initial dt = %.5f ...\n",
             t0,tf,y0,dtinitial); CHKERRQ(ierr);
  ierr = TSSolve(ts,y); CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

