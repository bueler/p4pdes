static char help[] = "Solve a 2-variable polynomial optimization problem.\n"
"Has FormObjective() and FormFunction() implementations.  The latter is the\n"
"residual and also the gradient of the objective.  The Jacobian (= Hessian of\n"
"objective) is not implemented.  Usage is either:\n"
"    ./cartoon -snes_[fd|mf]\n"
"or:\n"
"    ./cartoon -snes_[fd|mf] -snes_fd_function\n"
"Use this or similar to count FormObjective() and FormFunction() evaluations:\n"
"    ./cartoon -snes_fd -log_summary|grep Eval\n\n";

/*  RESULTS:

$ ./cartoon -snes_fd -snes_converged_reason -snes_rtol 1.0e-15
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 5
|x-x_exact|_inf = 0

$ ./cartoon -snes_mf -snes_converged_reason -snes_rtol 1.0e-15
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 5
|x-x_exact|_inf = 0

$ ./cartoon -snes_fd -snes_converged_reason -snes_rtol 1.0e-15 -snes_fd_function
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 4
|x-x_exact|_inf = 1.97337e-08

$ ./cartoon -snes_mf -snes_converged_reason -snes_rtol 1.0e-15 -snes_fd_function
Nonlinear solve converged due to CONVERGED_FNORM_ABS iterations 4
|x-x_exact|_inf = 5.97357e-08

*/

// NOTE:  this should work to report Function and Objective evaluations, but not yet in maint:
//    ./cartoon -snes_converged_reason -snes_rtol 1e-15 -snes_fd_function -snes_fd -log_summary |grep Eval

#include <petsc.h>

PetscErrorCode FormObjective(SNES snes, Vec x, PetscReal *Phi, void *ctx) {
    PetscErrorCode  ierr;
    const PetscReal *ax;

    ierr = VecGetArrayRead(x,&ax); CHKERRQ(ierr);
    *Phi = 0.25 * (pow(ax[0],4.0) + pow(ax[1],4.0)) - 2.0 * ax[0] + 2.0 * ax[1];
    ierr = VecRestoreArrayRead(x,&ax); CHKERRQ(ierr);
    return 0;
}

// residual F = grad Phi
PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    PetscErrorCode  ierr;
    const PetscReal *ax;
    PetscReal       *aF;

    ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
    aF[0] = ax[0]*ax[0]*ax[0] - 2.0;
    aF[1] = ax[1]*ax[1]*ax[1] + 2.0;
    ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    SNES           snes;          // nonlinear solver context
    Vec            x, r;          // soln and residual vectors
    const PetscInt ix[2] = {0,1};
    PetscReal      scale = 1.0, iv[2], err;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","cartoon options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-initialscale",
                   "scale initial vector x = [1 -1] by this value",
                   NULL,scale,&scale,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    iv[0] = 1.0 * scale;   iv[1] = -1.0 * scale;
    ierr = VecSetValues(x,2,ix,iv,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetObjective(snes,FormObjective,NULL); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,NULL); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);

    iv[0] = - pow(2.0,1.0/3.0);  iv[1] = - iv[0];  // negative of exact soln
    ierr = VecSetValues(x,2,ix,iv,ADD_VALUES); CHKERRQ(ierr); // note ADD_VALUES
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"|x-x_exact|_inf = %g\n",err); CHKERRQ(ierr);

    VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);
    PetscFinalize();
    return 0;
}

