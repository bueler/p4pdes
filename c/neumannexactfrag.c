PetscScalar lowfreq_exactsoln(PetscScalar x, PetscScalar y) {
  return cos(2.0*PETSC_PI*x) * cos(2.0*PETSC_PI*y);
}

PetscScalar lowfreq_f(PetscScalar x, PetscScalar y) {
  return 8.0 * PETSC_PI * PETSC_PI * lowfreq_exactsoln(x,y);
}

PetscScalar lowfreq_gamma(PetscScalar x, PetscScalar y) {
  return 0.0;
}

    // SOLVE HOMOGENEOUS NEUMANN PROBLEM WITH KNOWN SOLN
    // (EVALUATES EXACT SOLUTION AT NODES)
    Vec          u,uexact;
    PetscScalar  uval, *ax, *ay, uudot, normuexact, normerror;
    PetscInt     i, Istart,Iend;
    KSP          ksp;
    MatNullSpace nullsp;
    ierr = assemble(WORLD,E,&lowfreq_f,NULL,NULL,A,b); CHKERRQ(ierr);
    ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(uexact,&Istart,&Iend); CHKERRQ(ierr);
    ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
    for (i = Istart; i < Iend; i++) {
      uval = lowfreq_exactsoln(ax[i-Istart],ay[i-Istart]);
      ierr = VecSetValues(uexact,1,&i,&uval,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
    vecassembly(uexact)
    // NEXT SOLVE SYSTEM
    ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
    ierr = KSPCreate(WORLD, &ksp); CHKERRQ(ierr);
    // only constants are in null space:
    ierr = MatNullSpaceCreate(WORLD, PETSC_TRUE, 0, NULL, &nullsp); CHKERRQ(ierr);
    ierr = KSPSetNullSpace(ksp, nullsp); CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);
    // COMPUTE ERROR; NOTE NUMERICAL SOLUTION MUST BE SCALED BECAUSE OF NON-UNIQUENESS
    ierr = VecDot(u,uexact,&uudot); CHKERRQ(ierr);
    ierr = VecScale(u,1.0/uudot); CHKERRQ(ierr);
    ierr = VecDot(uexact,uexact,&uudot); CHKERRQ(ierr);
    ierr = VecScale(uexact,1.0/uudot); CHKERRQ(ierr);
    ierr = VecNorm(uexact,NORM_2,&normuexact); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u := -uexact + u
    ierr = VecNorm(u,NORM_2,&normerror); CHKERRQ(ierr);
    ierr = PetscPrintf(WORLD,"  solving homogenous: |u - uexact|_2 / |uexact|_2 = %e  (should be O(h^2))\n",
                     normerror/normuexact); CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    VecDestroy(&u);  VecDestroy(&uexact);
