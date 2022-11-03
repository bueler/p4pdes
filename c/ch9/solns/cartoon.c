static char help[] = "Solve a 2-variable polynomial optimization problem.\n"
"Implements an objective function and its gradient (the residual).  The \n"
"Jacobian (Hessian) is not implemented.  Usage is either objective-only\n"
"    ./cartoon -snes_[fd|mf] -snes_fd_function\n"
"or with gradient\n"
"    ./cartoon -snes_[fd|mf]\n\n";

#include <petsc.h>

PetscErrorCode FormObjective(SNES snes, Vec x, PetscReal *Phi, void *ctx) {
    const PetscReal *ax;

    PetscCall(VecGetArrayRead(x,&ax));
    *Phi = 0.25 * (pow(ax[0],4.0) + pow(ax[1],4.0)) - 2.0 * ax[0] + 2.0 * ax[1];
    PetscCall(VecRestoreArrayRead(x,&ax));
    return 0;
}

// residual F = grad Phi
PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    const PetscReal *ax;
    PetscReal       *aF;

    PetscCall(VecGetArrayRead(x,&ax));
    PetscCall(VecGetArray(F,&aF));
    aF[0] = ax[0]*ax[0]*ax[0] - 2.0;
    aF[1] = ax[1]*ax[1]*ax[1] + 2.0;
    PetscCall(VecRestoreArrayRead(x,&ax));
    PetscCall(VecRestoreArray(F,&aF));
    return 0;
}

int main(int argc,char **argv) {
    SNES           snes;
    Vec            x, r, xexact;
    const PetscInt ix[2] = {0,1};
    PetscReal      iv[2], err;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x,PETSC_DECIDE,2));
    PetscCall(VecSetFromOptions(x));

    iv[0] = 0.5;   iv[1] = 0.5;    // initial iterate corresponds to nonsingular Hessian
    PetscCall(VecSetValues(x,2,ix,iv,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetObjective(snes,FormObjective,NULL));
    PetscCall(VecDuplicate(x,&r));
    PetscCall(SNESSetFunction(snes,r,FormFunction,NULL));
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSolve(snes,NULL,x));

    iv[0] = pow(2.0,1.0/3.0);  iv[1] = - iv[0];  // exact soln
    PetscCall(VecDuplicate(x,&xexact));
    PetscCall(VecSetValues(xexact,2,ix,iv,ADD_VALUES)); // note ADD_VALUES
    PetscCall(VecAssemblyBegin(xexact));
    PetscCall(VecAssemblyEnd(xexact));
    PetscCall(VecAXPY(x,-1.0,xexact));    // x <-- x + (-1.0) xexact
    PetscCall(VecNorm(x,NORM_INFINITY,&err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"numerical error |x-x_exact|_inf = %g\n",err));

    VecDestroy(&x);  VecDestroy(&r);  VecDestroy(&xexact);
    SNESDestroy(&snes);
    PetscCall(PetscFinalize());
    return 0;
}
