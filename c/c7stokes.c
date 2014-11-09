
static char help[] = "Solves a structured-grid Stokes problem with DMDA and KSP.\n\n";

#include <petscdmda.h>
#include <petscsnes.h>


typedef struct {
  PetscReal u;
  PetscReal v;
  PetscReal p;
} Field;

typedef struct {
  DM        da;
  PetscInt  dof;  // number of degrees of freedom at each node
  PetscReal L,    // length of domain in x direction
            H;    // length of domain in y direction
} AppCtx;

extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,AppCtx*);
//extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,Field**,Mat,Mat,AppCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;                         /* user-defined work context */
  SNES           snes;                         /* nonlinear solver */
  Vec            x;                            /* solution vector */
  PetscInt       its;                          /* iterations for convergence */

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  user.dof = 3;
  user.L   = 10.0;
  user.H   = 1.0;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                      -4,-4,PETSC_DECIDE,PETSC_DECIDE,
                      user.dof,1,NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.H, 0.0, user.L, -1.0, -1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);

  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
                                  (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  //ierr = FormInitialGuess(da,&user,x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
