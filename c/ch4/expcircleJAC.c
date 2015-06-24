static char help[] =
   "Newton's method for a two-variable system with analytical Jacobian.\n\n";

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

//STARTJAC
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat J,Mat Jpre,void *dummy) {
  PetscErrorCode    ierr;
  const PetscReal   *ax;
  PetscScalar       v[4];
  PetscInt          row[2] = {0,1}, col[2] = {0,1};

  ierr = VecGetArrayRead(x,&ax); CHKERRQ(ierr);
  v[0] = PetscExpReal(ax[0]);  v[1] = -2.0;
  v[2] = 2.0 * ax[0];          v[3] = 2.0 * ax[1];
  ierr = MatSetValues(Jpre,2,row,2,col,v,INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&ax); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
//ENDJAC

int main(int argc,char **argv)
{
  SNES  snes;         // nonlinear solver context
  Vec   x,r;          // solution, residual vectors
  Mat   J;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r); CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,NULL);CHKERRQ(ierr);
//STARTADDJAC
  ierr = MatCreate(PETSC_COMM_WORLD,&J); CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,NULL);CHKERRQ(ierr);
//ENDADDJAC
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecSet(x,1.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&x);  VecDestroy(&r);  SNESDestroy(&snes);  MatDestroy(&J);
  PetscFinalize();
  return 0;
}

