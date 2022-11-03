static char help[] = "Newton's method for arctan x = 0.  Run with -snes_fd or -snes_mf.\n\n";

#include <petsc.h>

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    SNES      snes;          // nonlinear solver
    Vec       x, r;          // solution, residual vectors
    PetscReal x0 = 2.0;

    PetscInitialize(&argc,&argv,(char*)0,help);

    PetscOptionsBegin(PETSC_COMM_WORLD,"","options to atan","");
    PetscCall(PetscOptionsReal("-x0","initial value","atan.c",x0,&x0,NULL));
    PetscOptionsEnd();

    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x,PETSC_DECIDE,1));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSet(x,x0));
    PetscCall(VecDuplicate(x,&r));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetFunction(snes,r,FormFunction,NULL));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSolve(snes,NULL,x));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

    VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    const PetscReal  *ax;
    PetscReal        *aF;

    PetscCall(VecGetArrayRead(x,&ax));
    PetscCall(VecGetArray(F,&aF));
    aF[0] = atan(ax[0]);
    PetscCall(VecRestoreArrayRead(x,&ax));
    PetscCall(VecRestoreArray(F,&aF));
    return 0;
}
