static char help[] = "Newton's method for a two-variable system.  Implements analytical Jacobian and a struct to hold a parameter.\n";

#include <petsc.h>

typedef struct {
  PetscReal  b;
} AppCtx;

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void*);

//STARTMAIN
int main(int argc,char **argv) {
  SNES   snes;         // nonlinear solver
  Vec    x,r;          // solution, residual vectors
  Mat    J;
  AppCtx user;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  user.b = 2.0;

  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,2));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&r));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetFunction(snes,r,FormFunction,&user));
  PetscCall(SNESSetJacobian(snes,J,J,FormJacobian,&user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecSet(x,1.0));            // initial iterate
  PetscCall(SNESSolve(snes,NULL,x));
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}
//ENDMAIN

//STARTFORM
PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    AppCtx          *user = (AppCtx*)ctx;
    const PetscReal b = user->b, *ax;
    PetscReal       *aF;

    PetscCall(VecGetArrayRead(x,&ax));
    PetscCall(VecGetArray(F,&aF));
    aF[0] = (1.0 / b) * PetscExpReal(b * ax[0]) - ax[1];
    aF[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0;
    PetscCall(VecRestoreArrayRead(x,&ax));
    PetscCall(VecRestoreArray(F,&aF));
    return 0;
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx) {
    AppCtx           *user = (AppCtx*)ctx;
    const PetscReal  b = user->b, *ax;
    PetscReal        v[4];
    PetscInt         row[2] = {0,1}, col[2] = {0,1};

    PetscCall(VecGetArrayRead(x,&ax));
    v[0] = PetscExpReal(b * ax[0]);  v[1] = -1.0;
    v[2] = 2.0 * ax[0];              v[3] = 2.0 * ax[1];
    PetscCall(VecRestoreArrayRead(x,&ax));
    PetscCall(MatSetValues(P,2,row,2,col,v,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
//ENDFORM
