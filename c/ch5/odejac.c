static char help[] =
"ODE system solver example using TS, but with Jacobian.  Sets TS type to\n"
"implicit Crank-Nicolson.  Compare ode.c.\n\n";

#include <petsc.h>

extern PetscErrorCode ExactSolution(PetscReal, Vec);
extern PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode FormRHSJacobian(TS, PetscReal, Vec, Mat, Mat, void*);

int main(int argc,char **argv) {
  PetscInt   steps;
  PetscReal  t0 = 0.0, tf = 20.0, dt = 0.1, err;
  Vec        y, yexact;
  Mat        J;
  TS         ts;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,2));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecDuplicate(y,&yexact));

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL));

//STARTMATJ
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(TSSetRHSJacobian(ts,J,J,FormRHSJacobian,NULL));
  PetscCall(TSSetType(ts,TSCN));
//ENDMATJ

  // set time axis
  PetscCall(TSSetTime(ts,t0));
  PetscCall(TSSetMaxTime(ts,tf));
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  // set initial values and solve
  PetscCall(TSGetTime(ts,&t0));
  PetscCall(ExactSolution(t0,y));
  PetscCall(TSSolve(ts,y));

  // compute error and report
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(TSGetTime(ts,&tf));
  PetscCall(ExactSolution(tf,yexact));
  PetscCall(VecAXPY(y,-1.0,yexact));    // y <- y - yexact
  PetscCall(VecNorm(y,NORM_INFINITY,&err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "error at tf = %.3f with %d steps:  |y-y_exact|_inf = %g\n",
              tf,steps,err));

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&yexact));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ExactSolution(PetscReal t, Vec y) {
    PetscReal *ay;
    PetscCall(VecGetArray(y,&ay));
    ay[0] = t - PetscSinReal(t);
    ay[1] = 1.0 - PetscCosReal(t);
    PetscCall(VecRestoreArray(y,&ay));
    return 0;
}

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec y, Vec g,
                               void *ptr) {
    const PetscReal *ay;
    PetscReal       *ag;
    PetscCall(VecGetArrayRead(y,&ay));
    PetscCall(VecGetArray(g,&ag));
    ag[0] = ay[1];            // = g_1(t,y)
    ag[1] = - ay[0] + t;      // = g_2(t,y)
    PetscCall(VecRestoreArrayRead(y,&ay));
    PetscCall(VecRestoreArray(g,&ag));
    return 0;
}

//STARTJACOBIAN
PetscErrorCode FormRHSJacobian(TS ts, PetscReal t, Vec y, Mat J, Mat P,
                               void *ptr) {
    PetscInt   row[2] = {0, 1},  col[2] = {0, 1};
    PetscReal  v[4] = { 0.0, 1.0,
                       -1.0, 0.0};
    PetscCall(MatSetValues(P,2,row,2,col,v,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
//ENDJACOBIAN
