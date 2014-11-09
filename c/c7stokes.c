
static char help[] = "Solves a structured-grid Stokes problem with DMDA and KSP.\n\n";

#include <petscdmda.h>
#include <petscsnes.h>


typedef struct {
  PetscReal u;
  PetscReal v;
  PetscReal p;
} Field;


extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscScalar**,PetscScalar**,void*);
//extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar**,Mat,Mat,AppCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;                         /* nonlinear solver */
  Vec            x;                            /* solution vector */
  //AppCtx         user;                         /* user-defined work context */
  PetscInt       its;                          /* iterations for convergence */
  PetscReal      H, L;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                      -4,-4,
                      PETSC_DECIDE,PETSC_DECIDE,
                      1,1,NULL,NULL,&da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, H, 0.0, L, -1.0, -1.0);CHKERRQ(ierr);
  //ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                                  (DMDASNESFunction)FormFunctionLocal,NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  //ierr = FormInitialGuess(da,&user,x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
