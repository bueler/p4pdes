static char help[] = "Solves a 1D reaction-diffusion problem with DMDA and SNES.\n\n";

#include <petsc.h>

//SETUP
typedef struct {
  PetscReal L, rho, M, uLEFT, uRIGHT;
} AppCtx;

PetscErrorCode FormInitialGuess(DM da, AppCtx *user, Vec u) {
    PetscErrorCode ierr;
    PetscInt       i;
    PetscReal      L = user->L, h, x, *au;
    DMDALocalInfo  info;

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    h = L / (info.mx-1);
    ierr = DMDAVecGetArray(da,u,&au); CHKERRQ(ierr);
    for (i=info.xs; i<info.xs+info.xm; i++) {
        x = h * i;
        au[i] = user->uLEFT * (L - x)/L + user->uRIGHT * x/L;
    }
    ierr = DMDAVecRestoreArray(da,u,&au); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormExactSolution(DM da, AppCtx *user, Vec uex) {
    PetscErrorCode ierr;
    PetscInt       i;
    PetscReal      h, x, *auex;
    DMDALocalInfo  info;

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    h = user->L / (info.mx-1);
    ierr = DMDAVecGetArray(da,uex,&auex); CHKERRQ(ierr);
    for (i=info.xs; i<info.xs+info.xm; i++) {
        x = h * i;
        auex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    ierr = DMDAVecRestoreArray(da,uex,&auex); CHKERRQ(ierr);
    return 0;
}
//ENDSETUP

//FUNJAC
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u,
                                 PetscReal *f, AppCtx *user) {
    PetscInt       i;
    PetscReal      h = user->L / (info->mx-1), R;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            f[i] = u[i] - user->uLEFT;
        } else if (i == info->mx-1) {
            f[i] = u[i] - user->uRIGHT;
        } else {
            // generic interior location
            R = - user->rho * PetscSqrtReal(u[i]);
            f[i] = (u[i+1] - 2.0 * u[i] + u[i-1]) / h + h * R;
        }
    }
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar *u,
                                 Mat J, Mat Jpre, AppCtx *user) {
    PetscErrorCode ierr;
    PetscInt       i, col[3];
    PetscReal      h = user->L / (info->mx-1), dRdu, v[3];

    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            ierr = MatSetValues(Jpre,1,&i,1,&i,v,INSERT_VALUES); CHKERRQ(ierr);
        } else {
            dRdu = - (user->rho / 2.0) / PetscSqrtReal(u[i]);
            col[0] = i-1;  v[0] = 1.0 / h;
            col[1] = i;    v[1] = -2.0 / h + h * dRdu;
            col[2] = i+1;  v[2] = v[0];
            ierr = MatSetValues(Jpre,1,&i,3,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}
//ENDFUNJAC

//MAIN
int main(int argc,char **args) {
  PetscErrorCode      ierr;
  DM                  da;
  SNES                snes;
  AppCtx              user;
  Vec                 u, uexact;
  PetscReal           unorm, errnorm;
  SNESConvergedReason reason;
  DMDALocalInfo       info;

  PetscInitialize(&argc,&args,(char*)0,help);
  user.L      = 1.0;
  user.rho    = 10.0;
  user.M      = PetscSqr(user.rho / 12.0);
  user.uLEFT  = user.M;
  user.uRIGHT = user.M * PetscPowReal(user.L + 1.0,4.0);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,-9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,user.L,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = FormInitialGuess(da,&user,u); CHKERRQ(ierr);
  ierr = FormExactSolution(da,&user,uexact); CHKERRQ(ierr);

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
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
              "on %d point grid:  %s with |u-u_exact|_inf/|u|_inf = %g\n",
              info.mx,SNESConvergedReasons[reason],errnorm/unorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

