static char help[] = "Solves a 1D reaction-diffusion problem with DMDA and SNES.\n\n";

#include <petsc.h>

//SETUP
typedef struct {
  PetscReal rho, M, uLEFT, uRIGHT;
} AppCtx;

PetscErrorCode InitialAndExact(DM da, DMDALocalInfo *info, Vec u0, Vec uex,
                               AppCtx *user) {
    PetscErrorCode ierr;
    PetscInt  i;
    PetscReal h = 1.0 / (info->mx-1), x, *au0, *auex;
    ierr = DMDAVecGetArray(da,u0,&au0); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,uex,&auex); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        au0[i]  = user->uLEFT * (1.0 - x) + user->uRIGHT * x;
        auex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    ierr = DMDAVecRestoreArray(da,u0,&au0); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,uex,&auex); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u,
                                 PetscReal *f, AppCtx *user) {
    PetscInt  i;
    PetscReal h = 1.0 / (info->mx-1), R;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            f[i] = u[i] - user->uLEFT;
        } else if (i == info->mx-1) {
            f[i] = u[i] - user->uRIGHT;
        } else {  // interior location
            R = - user->rho * PetscSqrtReal(u[i]);
            f[i] = u[i+1] - 2.0 * u[i] + u[i-1] + h*h * R;
        }
    }
    return 0;
}
//ENDSETUP

//FUNJAC
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal *u,
                                 Mat J, Mat P, AppCtx *user) {
    PetscErrorCode ierr;
    PetscInt  i, col[3];
    PetscReal h = 1.0 / (info->mx-1), dRdu, v[3];
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            ierr = MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES); CHKERRQ(ierr);
        } else {
            dRdu = - (user->rho / 2.0) / PetscSqrtReal(u[i]);
            col[0] = i-1;  v[0] = 1.0;
            col[1] = i;    v[1] = -2.0 + h*h * dRdu;
//            col[1] = i;    v[1] = -2.0; //STRIP
            col[2] = i+1;  v[2] = v[0];
            ierr = MatSetValues(P,1,&i,3,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}
//ENDFUNJAC

//MAIN
int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM                  da;
  SNES                snes;
  AppCtx              user;
  Vec                 u, uexact;
  PetscReal           unorm, errnorm;
  DMDALocalInfo       info;

  PetscInitialize(&argc,&args,(char*)0,help);
  user.rho    = 10.0;
  user.M      = PetscSqr(user.rho / 12.0);
  user.uLEFT  = user.M;
  user.uRIGHT = 16.0 * user.M;

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = InitialAndExact(da,&info,u,uexact,&user); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
              "on %d point grid:  |u-u_exact|_inf/|u|_inf = %g\n",
              info.mx,errnorm/unorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

