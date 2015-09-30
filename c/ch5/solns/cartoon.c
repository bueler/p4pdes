static char help[] = "Solve a 2-variable polynomial optimization problem.\n"
"When using -objonly option, only seems to work this way:\n"
"    ./cartoon -objonly -snes_fd_function -snes_mf\n"
"To use residual (i.e. gradient-of-objective) evaluation routine, plus\n"
"objective function in the line search, run as\n"
"    ./cartoon -snes_fd\n\n";

#include <petsc.h>

// the exact minimum of this objective is  Phi(2^(1/3),-2^(1/3)) = - 3 * 2^(1/3)
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
    PetscErrorCode   ierr;
    const PetscReal  *ax;
    PetscReal        *aF;

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
    Vec            x, r;          // solution vector
    PetscReal      initial = 1.0;
    PetscBool      objonly = PETSC_FALSE;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","cartoon options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-initial","initial vector has this value for both components",
                   NULL,initial,&initial,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-objonly","do not use residual evaluation routine at all",
                   NULL,objonly,&objonly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,"-initial",&initial,NULL); CHKERRQ(ierr);
    ierr = VecSet(x,initial); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetObjective(snes,FormObjective,NULL); CHKERRQ(ierr);
    if (!objonly) {
        ierr = SNESSetFunction(snes,r,FormFunction,NULL); CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);
    PetscFinalize();
    return 0;
}

