
static char help[] = "Minimal test of linear ODE solve by TS.  Solves dy/dt = y with y(0) = 1 to compute y(1).  Answer should be y(1)=e, but can't seem to force final time to be tf=1.0.\n";

#include <petscts.h>

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  PetscInitialize(&argc,&argv,(char*)0,help);

  PetscBool wanttfexact = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for testts", ""); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-want_tf_exact",
                          "do call to TSSetExactFinalTime()\n", "", wanttfexact,
                          &wanttfexact, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  Vec x;
  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,1); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecSet(x,1.0); CHKERRQ(ierr);

  Mat  A;
  ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,1,1); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  PetscInt    i,Istart,Iend;
  PetscScalar v;
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRQ(ierr);
  for (i=Istart; i<Iend; i++) {
    v = 1.0;
    ierr = MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // CREATE AND SETUP TS
  TS  ts;
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,NULL);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,A,A,TSComputeRHSJacobianConstant,NULL);CHKERRQ(ierr);

  // SET INITIAL CONDITION AND AS FOR DURATION
  PetscReal t0 = 0.0, tf = 1.0, dtinitial = 0.1;
  ierr = TSSetInitialTimeStep(ts,t0,dtinitial);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100,tf);CHKERRQ(ierr);

  //FIXME: seems to have no effect?
  if (wanttfexact == PETSC_TRUE) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,
             "calling TSSetExactFinalTime() ...\n"); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "from t0 = %g to tf = %g, with dt0 = %f ...\n",
             t0,tf,dtinitial); CHKERRQ(ierr);

  // SOLVE (USING USER'S OPTIONS)
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  // show result:
  // FIXME: does not return tf, as promised in man page for TSGetSolveTime()
  PetscReal tfreturned;
  ierr = TSGetSolveTime(ts,&tfreturned);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "result x at t = %g (note |tf-tr_returned| = %e):\n",
             tfreturned,fabs(tf-tfreturned)); CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

