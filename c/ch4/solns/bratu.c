static char help[] = "Solves the Bratu reaction-diffusion problem in 1D.\n"
"Optionally includes manufactured solutio to generalized problem.\n";

#include <petsc.h>

typedef struct {
  PetscBool manufactured;
  PetscReal lambda;
} AppCtx;

PetscErrorCode Initial(DM da, DMDALocalInfo *info, Vec u0, AppCtx *user) {
    PetscErrorCode ierr;
    PetscInt  i;
    PetscReal *au0;
    ierr = DMDAVecGetArray(da,u0,&au0); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        au0[i] = 0.0;
    }
    ierr = DMDAVecRestoreArray(da,u0,&au0); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode Exact(DM da, DMDALocalInfo *info, Vec uex, AppCtx *user) {
    PetscErrorCode ierr;
    PetscInt  i;
    PetscReal h = 1.0 / (info->mx-1), x, *auex;
    ierr = DMDAVecGetArray(da,uex,&auex); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = i * h;
        auex[i] = sin(PETSC_PI * x);
    }
    ierr = DMDAVecRestoreArray(da,uex,&auex); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u,
                                 PetscReal *f, AppCtx *user) {
    PetscInt  i;
    PetscReal h = 1.0 / (info->mx-1), x, R, compf;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) || (i == info->mx-1)) {
            f[i] = u[i];
        } else {  // interior location
            R = user->lambda * PetscExpReal(u[i]);
            f[i] = - u[i+1] + 2.0 * u[i] - u[i-1] - h*h * R;
            if (user->manufactured) {
                PetscReal uex;
                x = i * h;
                uex = sin(PETSC_PI * x);
                compf = PETSC_PI * PETSC_PI * uex - user->lambda * PetscExpReal(uex);
                f[i] -= h * h * compf;
            }
        }
    }
    return 0;
}

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
            dRdu = user->lambda * PetscExpReal(u[i]);
            col[0] = i-1;  v[0] = - 1.0;
            col[1] = i;    v[1] = 2.0 - h*h * dRdu;
            col[2] = i+1;  v[2] = - 1.0;
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

int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM                  da;
  SNES                snes;
  AppCtx              user;
  Vec                 u;
  DMDALocalInfo       info;

  PetscInitialize(&argc,&args,(char*)0,help);
  user.lambda = 1.0;
  user.manufactured = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,
                           "bra_","options for bratu",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda","coefficient of nonlinear zeroth-order term",
                          "bratu.c",user.lambda,&(user.lambda),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-manufactured","if set, use a manufactured solution",
                          "bratu.c",user.manufactured,&(user.manufactured),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
  ierr = Initial(da,&info,u,&user); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  if (user.manufactured) {
      Vec       uexact;
      PetscReal unorm, errnorm;
      ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
      ierr = Exact(da,&info,uexact,&user); CHKERRQ(ierr);
      ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
              "on %d point grid:  |u-u_exact|_inf/|u|_inf = %g\n",
              info.mx,errnorm/unorm); CHKERRQ(ierr);
      VecDestroy(&uexact);
  } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"done on %d point grid\n",info.mx); CHKERRQ(ierr);
  }

  VecDestroy(&u);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

