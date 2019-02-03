static char help[] = "Newton's method for a three-variable system.\n\n";

#include <petsc.h>

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    SNES  snes;          // nonlinear solver context
    Vec   x, r;          // solution, residual vectors
    double *ax;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
    ierr = VecSetSizes(x,PETSC_DECIDE,3); CHKERRQ(ierr);
    ierr = VecSetFromOptions(x); CHKERRQ(ierr);
    ierr = VecSetUp(x); CHKERRQ(ierr);
    ierr = VecGetArray(x,&ax);CHKERRQ(ierr);
    // ax = [N, A, s]
    ax[0] = 200.0;
    ax[1] = 1.0;
    ax[2] = 0.5;
    ierr = VecRestoreArray(x,&ax);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,r,FormFunction,NULL); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

    VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);
    return PetscFinalize();
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    PetscErrorCode ierr;
    const double *ax;
    double       *aF;

    ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
    // ax = [N, A, s]
    // 3.929 (1 + A)         - N = 0
    // 23.192 (1 + A s^N)    - N = 0
    // 91.972 (1 + A s^(2N)) - N = 0
    aF[0] = 3.929 * (1.0 + ax[1]) - ax[0];
    aF[1] = 23.192 * (1.0 + ax[1] * pow(ax[2],ax[1])) - ax[0];
    aF[2] = 91.972 * (1.0 + ax[1] * pow(ax[2],2.0*ax[1])) - ax[0];
    ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
    return 0;
}


