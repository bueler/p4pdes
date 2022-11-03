static char help[] = "Newton's method for a three-variable system.\n"
"The system is from fitting logistic population model\n"
"P(t) = N / (1 + A exp(-Nbt)) to 3 points (t,P) from census data in\n"
"D. Zill, 11th ed., exercise 4 in section 3.2.  The points are\n"
"(1790,3.929), (1850,23.192), (1910,91.972); populations in millions.\n"
"Run with -snes_fd.\n";

/*
example:
$ make census
$ ./census -snes_fd -snes_monitor
  0 SNES Function norm 2.233793904584e+01
...
 17 SNES Function norm 6.521724317021e-07
 18 SNES Function norm 4.358557709045e-09
Vec Object: 1 MPI processes
  type: seq
197.274
49.2096
0.000158863
*/

#include <petsc.h>

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    SNES      snes;          // nonlinear solver
    Vec       x, r;          // solution, residual vectors
    PetscReal *ax;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
    PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
    PetscCall(VecSetSizes(x,PETSC_DECIDE,3));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSetUp(x));
    PetscCall(VecGetArray(x,&ax));
    // ax = [N, A, b]; see equations in FormFunction() below
    // these initial values are based on *guessing* N=300
    // as the limiting population, then computing A, then b
    ax[0] = 300.0;
    ax[1] = 75.0;
    ax[2] = 1.0e-4;
    PetscCall(VecRestoreArray(x,&ax));
    PetscCall(VecAssemblyBegin(x));
    PetscCall(VecAssemblyEnd(x));
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
    const PetscReal *ax;
    PetscReal       *aF;

    PetscCall(VecGetArrayRead(x,&ax));
    PetscCall(VecGetArray(F,&aF));
    // ax = [N, A, b]
    //  3.929 (1 + A)               - N = 0
    // 23.192 (1 + A exp(-60*N*b))  - N = 0
    // 91.972 (1 + A exp(-120*N*b)) - N = 0
    aF[0] =  3.929 * (1.0 + ax[1]) - ax[0];
    aF[1] = 23.192 * (1.0 + ax[1] * PetscExpReal(-60.0*ax[0]*ax[2])) - ax[0];
    aF[2] = 91.972 * (1.0 + ax[1] * PetscExpReal(-120.0*ax[0]*ax[2])) - ax[0];
    PetscCall(VecRestoreArray(F,&aF));
    PetscCall(VecRestoreArrayRead(x,&ax));
    return 0;
}

