static char help[] =
"ODE system solver example using TS.  Option prefix -ode_.\n"
"Solves N-dimensional system  dy/dt = G(t,y)  with y(t0) = y0 to compute y(tf),\n"
"where t0, tf are set by options but y0 must be set in code.  Serial only.\n"
"Defaults to Runge-Kutta (= 3BS), but can be used with implicit methods.\n"
"The implemented example, which includes the Jacobian  J = dG/dy,  has\n"
"G_1 = y_2, G_2 = - y_1 + t, y_1(0) = 0, y_2(0) = 0.  The exact solution is\n"
"y_1(t) = t - sin(t), y_2(t) = 1 - cos(t).\n\n";

// ./ode -ts_monitor
// ./ode -ts_monitor_solution
// ./ode -ts_monitor_solution draw -draw_pause 0.1

// ./ode -ts_monitor -ts_type beuler -ode_steps 1000    // finally close to default RK
// ./ode -log_view |grep Eval   // compare rk, beuler, cn

// ./ode -ts_type euler   // time-stepping failure; see petsc issue #119
// ./ode -ts_monitor -ts_type rk -ts_rk_type 1fe -ts_adapt_type none   // correct Euler

#include <petsc.h>

PetscErrorCode SetInitial(Vec y) {
    double *ay;
    VecGetArray(y,&ay);
    ay[0] = 0.0;
    ay[1] = 0.0;
    VecRestoreArray(y,&ay);
    PetscFunctionReturn(0);
}

PetscErrorCode FormRHSFunction(TS ts, double t, Vec y, Vec g, void *ptr) {
    const double *ay;
    double       *ag;
    VecGetArrayRead(y,&ay);
    VecGetArray(g,&ag);
    ag[0] = ay[1];            // = G_1(t,y)
    ag[1] = - ay[0] + t;      // = G_2(t,y)
    VecRestoreArrayRead(y,&ay);
    VecRestoreArray(g,&ag);
    PetscFunctionReturn(0);
}

PetscErrorCode FormRHSJacobian(TS ts, double t, Vec y, Mat J, Mat P, void *ptr) {
    PetscErrorCode ierr;
    const double *ay;
    double       v[4] = { 0.0, 1.0,
                         -1.0, 0.0};
    int          row[2] = {0, 1},  col[2] = {0, 1};
    ierr = VecGetArrayRead(y,&ay); CHKERRQ(ierr);
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
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
  const int N = 2;
  double    t0 = 0.0,  tf = 1.0,  dt = 0.1;
  Vec       y, yexact;
  Mat       J;
  TS        ts;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ode_", 
                           "options for ODE solver ode.c", ""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","initial time-step","ode.c",dt,&dt,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-t0","initial time","ode.c",t0,&t0,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","final time","ode.c",tf,&tf,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y); CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N); CHKERRQ(ierr);
  ierr = VecSetFromOptions(y); CHKERRQ(ierr);
  ierr = VecDuplicate(y,&yexact); CHKERRQ(ierr);
  ierr = SetInitialExact(t0,y,tf,yexact); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL); CHKERRQ(ierr);

  ierr = TSSetInitialTimeStep(ts,t0,dt); CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100*(int)(tf/dt),tf-t0); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "solving t0 = %.3f to tf = %.3f with initial dt = %.5f ...\n",
             t0,tf,dt); CHKERRQ(ierr);
  ierr = TSSolve(ts,y); CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&y); VecDestroy(&yexact);
  MatDestroy(&J); TSDestroy(&ts);
  PetscFinalize();
  return 0;
}

