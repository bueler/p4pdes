static char help[] = "Solve a 2-variable polynomial optimization problem.\n"
"Implements an objective function and its gradient (the residual).  The \n"
"Jacobian (Hessian) is not implemented.  Usage is either objective-only\n"
"    ./cartoon -snes_[fd|mf] -snes_fd_function\n"
"or with gradient\n"
"    ./cartoon -snes_[fd|mf]\n\n";

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
    SNES           snes;
    Vec            x, r, xexact;
    const PetscInt ix[2] = {0,1};
    PetscReal      iv[2], err;

    PetscInitialize(&argc,&argv,NULL,help);

    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);

    iv[0] = 0.5;   iv[1] = 0.5;    // initial iterate corresponds to nonsingular Hessian
    ierr = VecSetValues(x,2,ix,iv,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetObjective(snes,FormObjective,NULL); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,NULL); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);

    iv[0] = pow(2.0,1.0/3.0);  iv[1] = - iv[0];  // exact soln
    ierr = VecDuplicate(x,&xexact); CHKERRQ(ierr);
    ierr = VecSetValues(xexact,2,ix,iv,ADD_VALUES); CHKERRQ(ierr); // note ADD_VALUES
    ierr = VecAssemblyBegin(xexact); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xexact); CHKERRQ(ierr);
    ierr = VecAXPY(x,-1.0,xexact); CHKERRQ(ierr);    // x <-- x + (-1.0) xexact
    ierr = VecNorm(x,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"numerical error |x-x_exact|_inf = %g\n",err); CHKERRQ(ierr);

    VecDestroy(&x);  VecDestroy(&r);  VecDestroy(&xexact);
    SNESDestroy(&snes);
    PetscFinalize();
    return 0;
}

