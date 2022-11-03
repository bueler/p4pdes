static char help[] = "Solves the Liouville-Bratu reaction-diffusion problem in 1D:\n"
"    - u'' - lambda e^u = 0\n"
"on [0,1] subject to homogeneous Dirichlet boundary conditions.  Optionally uses manufactured solution to problem with f(x) on right-hand-side.\n\n";

#include <petsc.h>

typedef struct {
    PetscBool  manufactured;
    PetscReal  lambda;
} AppCtx;

extern PetscErrorCode Exact(DM, DMDALocalInfo*, Vec, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*,
                                 PetscReal*, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscReal*,
                                 Mat, Mat, AppCtx*);

int main(int argc,char **args) {
  DM                  da;
  SNES                snes;
  AppCtx              user;
  Vec                 u, uexact;
  PetscReal           unorm, errnorm;
  DMDALocalInfo       info;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  user.lambda = 1.0;
  user.manufactured = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","options for bratu1D","");
  PetscCall(PetscOptionsReal("-lambda","coefficient of nonlinear zeroth-order term",
                          "bratu1D.c",user.lambda,&(user.lambda),NULL));
  PetscCall(PetscOptionsBool("-manu","if set, use a manufactured solution",
                          "bratu1D.c",user.manufactured,&(user.manufactured),NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(da,&u));
  PetscCall(VecSet(u,0.0));
  PetscCall(SNESSolve(snes,NULL,u));

  PetscCall(DMDAGetLocalInfo(da,&info));
  if (user.manufactured) {
      PetscCall(VecDuplicate(u,&uexact));
      PetscCall(Exact(da,&info,uexact,&user));
      PetscCall(VecNorm(u,NORM_INFINITY,&unorm));
      PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uxact
      PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
              "on %d point grid:  |u-u_exact|_inf/|u|_inf = %g\n",
              info.mx,errnorm/unorm));
      VecDestroy(&uexact);
  } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d point grid\n",info.mx));
  }

  VecDestroy(&u);  SNESDestroy(&snes);  DMDestroy(&da);
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode Exact(DM da, DMDALocalInfo *info, Vec uex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x, *auex;
    PetscCall(DMDAVecGetArray(da,uex,&auex));
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = i * h;
        auex[i] = sin(PETSC_PI * x);
    }
    PetscCall(DMDAVecRestoreArray(da,uex,&auex));
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u,
                                 PetscReal *f, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x, R, compf, uex;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) || (i == info->mx-1)) {
            f[i] = u[i];
        } else {  // interior location
            R = user->lambda * PetscExpReal(u[i]);
            f[i] = - u[i+1] + 2.0 * u[i] - u[i-1] - h*h * R;
            if (user->manufactured) {
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
    PetscInt   i, col[3];
    PetscReal  h = 1.0 / (info->mx-1), dRdu, v[3];
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            PetscCall(MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES));
        } else {
            dRdu = user->lambda * PetscExpReal(u[i]);
            col[0] = i-1;  v[0] = - 1.0;
            col[1] = i;    v[1] = 2.0 - h*h * dRdu;
            col[2] = i+1;  v[2] = - 1.0;
            PetscCall(MatSetValues(P,1,&i,3,col,v,INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}

