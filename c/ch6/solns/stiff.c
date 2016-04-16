static char help[] =
"Siff ODE system.  Compare odejac.c.\n\n";

#include <petsc.h>

PetscErrorCode FormRHSFunction(TS ts, double t, Vec y, Vec g, void *ptr) {
    const double *ay;
    double       *ag;
    VecGetArrayRead(y,&ay);
    VecGetArray(g,&ag);
    ag[0] = ay[1];
    ag[1] = - ay[0];
    ag[2] = - 200.0 * ay[2];
    VecRestoreArrayRead(y,&ay);
    VecRestoreArray(g,&ag);
    return 0;
}

PetscErrorCode FormRHSJacobian(TS ts, double t, Vec y, Mat J, Mat P,
                               void *ptr) {
    PetscErrorCode ierr;
    int    j[3] = {0, 1, 2};
    double v[9] = { 0.0, 1.0, 0.0,
                   -1.0, 0.0, 0.0,
                    0.0, 0.0, -200.0};
    ierr = MatSetValues(P,3,j,3,j,v,INSERT_VALUES); CHKERRQ(ierr);
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
  const int N = 3;
  Vec       y;
  Mat       J;
  TS        ts;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = VecCreate(PETSC_COMM_WORLD,&y); CHKERRQ(ierr);
  ierr = VecSetSizes(y,PETSC_DECIDE,N); CHKERRQ(ierr);
  ierr = VecSetFromOptions(y); CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL); CHKERRQ(ierr);

  ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,1.0); CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100000,10.0); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);  // can override defaults

  ierr = VecSet(y,1.0); CHKERRQ(ierr);
  ierr = TSSolve(ts,y); CHKERRQ(ierr);

  VecDestroy(&y);  TSDestroy(&ts);  MatDestroy(&J);
  PetscFinalize();
  return 0;
}

