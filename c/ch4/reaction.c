static char help[] = "Solves a 1D reaction-diffusion problem with DMDA and SNES.\n\n";

#include <petsc.h>

//SETUP
typedef struct {
  PetscReal L, D, rho, uLEFT, uRIGHT;
} AppCtx;

PetscErrorCode FormInitialGuess(DM da, AppCtx *user, Vec u) {
    PetscErrorCode ierr;
    PetscInt       i;
    PetscReal      L = user->L, h, x, *au;
    DMDALocalInfo  info;

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    h = L / (info.mx-1);
    ierr = DMDAVecGetArray(da,u,&au);CHKERRQ(ierr);
    for (i=info.xs; i<info.xs+info.xm; i++) {
        x = h * i;
        au[i] = user->uLEFT * (L - x)/L + user->uRIGHT * x/L;
    }
    ierr = DMDAVecRestoreArray(da,u,&au);CHKERRQ(ierr);
    return 0;
}
//ENDSETUP

//FUNCTION
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u, PetscReal *f, AppCtx *user) {
    PetscInt       i;
    PetscReal      h = user->L / (info->mx-1), R;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            f[i] = u[i] - user->uLEFT;
        } else if (i == info->mx-1) {
            f[i] = u[i] - user->uRIGHT;
        } else {
            // generic interior location case
            R = user->rho * u[i] * (1.0 - u[i]);
            f[i] = user->D * (u[i+1] - 2.0 * u[i] + u[i-1]) / h + h * R;
        }
    }
    return 0;
}
//ENDFUNCTION

//extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,PetscScalar*,Mat,Mat,AppCtx*);

//MAIN
int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  AppCtx         user;
  Vec            u;
  PetscInt       iter;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&args,(char*)0,help);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  user.L = 1.0;
  user.D = 1.0;
  user.rho = 100.0;
  user.uLEFT = 0.0;
  user.uRIGHT = 1.0;
  ierr = FormInitialGuess(da,&user,u); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
//  ierr = DMDASNESSetJacobianLocal(da,
//             (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "on %d point grid:  %d Newton iterations\n",
              info.mx,iter); CHKERRQ(ierr);

  VecDestroy(&u);  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

