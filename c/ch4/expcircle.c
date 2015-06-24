//START
static char help[] = "Newton's method for a two-variable system.\n"
    "No analytical Jacobian, so run with -snes_fd or -snes_mf.\n\n";

#include <petsc.h>

PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx) {
  PetscErrorCode    ierr;
  const PetscReal   *ax;
  PetscReal         *af;

  ierr = VecGetArrayRead(x,&ax);CHKERRQ(ierr);
  ierr = VecGetArray(f,&af);CHKERRQ(ierr);
  af[0] = PetscExpReal(ax[0]) - 2.0 * ax[1];
  af[1] = ax[0] * ax[0] + ax[1] * ax[1] - 1.0;
  ierr = VecRestoreArrayRead(x,&ax);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&af);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  SNES  snes;         // nonlinear solver context
  Vec   x,r;          // solution, residual vectors
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,NULL); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = VecSet(x,1.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);
  PetscFinalize();
  return 0;
}
//END
