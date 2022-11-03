static char help[] = "Stiff ODE system.  Compare odejac.c.\n\n";

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

extern PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode FormRHSJacobian(TS, PetscReal, Vec, Mat, Mat, void*);

int main(int argc,char **argv) {
  const PetscInt N = 3;
  PetscInt       steps;
  Vec            y;
  Mat            J;
  TS             ts;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(y));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL));
  PetscCall(TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL));

  PetscCall(TSSetType(ts,TSRK));
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetMaxTime(ts,10.0));
  PetscCall(TSSetTimeStep(ts,1.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));  // can override defaults

  PetscCall(VecSet(y,1.0));
  PetscCall(TSSolve(ts,y));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "total steps = %d\n",steps));

  VecDestroy(&y);  TSDestroy(&ts);  MatDestroy(&J);
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec y, Vec g, void *ptr) {
    const PetscReal *ay;
    PetscReal       *ag;
    PetscCall(VecGetArrayRead(y,&ay));
    PetscCall(VecGetArray(g,&ag));
    ag[0] = ay[1];
    ag[1] = - ay[0] + 0.1 * ay[2];
    ag[2] = - 101.0 * ay[2];
    PetscCall(VecRestoreArrayRead(y,&ay));
    PetscCall(VecRestoreArray(g,&ag));
    return 0;
}

PetscErrorCode FormRHSJacobian(TS ts, PetscReal t, Vec y, Mat J, Mat P,
                               void *ptr) {
    PetscInt   j[3] = {0, 1, 2};
    PetscReal  v[9] = { 0.0, 1.0, 0.0,
                       -1.0, 0.0, 0.1,
                        0.0, 0.0, -101.0};
    PetscCall(MatSetValues(P,3,j,3,j,v,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
