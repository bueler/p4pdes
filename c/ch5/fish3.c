static char help[] = "Solves a 3D structured-grid Poisson problem with DMDA\n"
"and SNES.\n\n";

// compare fish2.c

#include <petsc.h>

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal ***x,
                                 PetscReal ***F, void *user) {
    SETERRQ(PETSC_COMM_WORLD,1,"NOT IMPLEMENTED");
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***x,
                                 Mat J, Mat Jpre, void *user) {
    SETERRQ(PETSC_COMM_WORLD,1,"NOT IMPLEMENTED");
    return 0;
}

//MAIN
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  Vec            u, uexact;
  PetscReal      errnorm;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,
                -5,-5,-5,
                PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                1,1,
                NULL,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,
            NULL);CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
            (PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,
            NULL);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
// FIXME  ierr = formInitialIterate(da,u); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
// FIXME  ierr = formExact(da,uexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d x %d x %d grid:  error |u-uexact|_inf = %g\n",
             info.mx,info.my,info.mz,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
//ENDMAIN

