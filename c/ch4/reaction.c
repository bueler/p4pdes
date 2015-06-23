static char help[] = "Solves a 1D reaction-diffusion problem with DMDA and SNES.\n\n";

#include <petsc.h>

typedef struct {
  PetscReal D, u_left, u_right;
} AppCtx;

extern PetscErrorCode FormInitialGuess(DM,AppCtx*,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar*,PetscScalar*,AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar*,Mat,Mat,AppCtx*);

int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  AppCtx         user;
  Vec            u,uexact;
  PetscInt       iter;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,
               -9,1,1,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = FormInitialGuess(da,&user,u); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d point grid:  %d iterations\n",
             info.mx,iter); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

