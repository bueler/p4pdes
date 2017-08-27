static char help[] =
"Siff ODE system.  Compare odejac.c.\n\n";

#include <petsc.h>

/* PYTHON:
import numpy as np
from scipy.linalg import expm
B = np.array([[0, 1, 0], [-1, 0, 0.1], [0, 0, -101]])
y = np.dot(expm(10*B),np.array([1.0,1.0,1.0]).transpose())
print y
RESULT:
[-1.383623   -0.29588643  0.        ]
*/

PetscErrorCode FormRHSFunction(TS ts, double t, Vec y, Vec g, void *ptr) {
    const double *ay;
    double       *ag;
    VecGetArrayRead(y,&ay);
    VecGetArray(g,&ag);
    ag[0] = ay[1];
    ag[1] = - ay[0] + 0.1 * ay[2];
    ag[2] = - 101.0 * ay[2];
    VecRestoreArrayRead(y,&ay);
    VecRestoreArray(g,&ag);
    return 0;
}

PetscErrorCode FormRHSJacobian(TS ts, double t, Vec y, Mat J, Mat P,
                               void *ptr) {
    PetscErrorCode ierr;
    int    j[3] = {0, 1, 2};
    double v[9] = { 0.0, 1.0, 0.0,
                   -1.0, 0.0, 0.1,
                    0.0, 0.0, -101.0};
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
  int       steps;
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
  ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10.0); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);  // can override defaults

  ierr = VecSet(y,1.0); CHKERRQ(ierr);
  ierr = TSSolve(ts,y); CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
              "total steps = %d\n",steps); CHKERRQ(ierr);

  VecDestroy(&y);  TSDestroy(&ts);  MatDestroy(&J);
  PetscFinalize();
  return 0;
}

