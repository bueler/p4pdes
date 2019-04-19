static char help[] =
   "Newton's method for a two-variable system.  Implements analytical Jacobian.\n"
   "Adds struct to hold parameter.\n";

#include <petsc.h>

typedef struct {
  double  b;
} AppCtx;

extern PetscErrorCode FormFunction(SNES, Vec, Vec, void*);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void*);

//STARTMAIN
int main(int argc,char **argv) {
  SNES   snes;         // nonlinear solver context
  Vec    x,r;          // solution, residual vectors
  Mat    J;
  AppCtx user;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,NULL,help);
  user.b = 2.0;

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);  MatDestroy(&J);
  return PetscFinalize();
}
//ENDMAIN

//STARTFORM
PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void *ctx) {
    PetscErrorCode ierr;
    AppCtx       *user = (AppCtx*)ctx;
    const double b = user->b, *ax;
    double       *aF;

    ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF);CHKERRQ(ierr);
    aF[0] = (1.0 / b) * PetscExpReal(b * ax[0]) - ax[1];
    aF[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0;
    ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx) {
    PetscErrorCode ierr;
    AppCtx       *user = (AppCtx*)ctx;
    const double b = user->b, *ax;
    double       v[4];
    int          row[2] = {0,1}, col[2] = {0,1};

    ierr = VecGetArrayRead(x,&ax); CHKERRQ(ierr);
    v[0] = PetscExpReal(b * ax[0]);  v[1] = -1.0;
    v[2] = 2.0 * ax[0];              v[3] = 2.0 * ax[1];
    ierr = VecRestoreArrayRead(x,&ax); CHKERRQ(ierr);
    ierr = MatSetValues(P,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}
//ENDFORM

