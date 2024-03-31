static char help[] = "1D reaction-diffusion problem with DMDA and SNES.  Option prefix -rct_.\n\n";

#include <petsc.h>

typedef struct {
    PetscReal  rho, M, alpha, beta;
    PetscBool  noRinJ;
} AppCtx;

extern PetscReal f_source(PetscReal);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*, PetscReal*, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscReal*, Mat, Mat, AppCtx*);

//STARTMAIN
int main(int argc,char **args) {
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           u, uexact;
  PetscReal     errnorm, *au, *auex;
  DMDALocalInfo info;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  user.rho   = 10.0;
  user.M     = PetscSqr(user.rho / 12.0);
  user.alpha = user.M;
  user.beta  = 16.0 * user.M;
  user.noRinJ = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_WORLD,"rct_","options for reaction",""); 
  PetscCall(PetscOptionsBool("-noRinJ","do not include R(u) term in Jacobian",
      "reaction.c",user.noRinJ,&(user.noRinJ),NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(DMCreateGlobalVector(da,&u));
  PetscCall(VecDuplicate(u,&uexact));
  PetscCall(DMDAVecGetArray(da,u,&au));

  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArray(da,uexact,&auex));
  PetscCall(InitialAndExact(&info,au,auex,&user));
  PetscCall(DMDAVecRestoreArray(da,u,&au));
  PetscCall(DMDAVecRestoreArray(da,uexact,&auex));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunctionFn *)FormFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobianFn *)FormJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes,NULL,u));

  PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
  PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |u-u_exact|_inf = %g\n",info.mx,errnorm));

  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&uexact));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
//ENDMAIN

PetscReal f_source(PetscReal x) {
    return 0.0;
}

PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0,
                               PetscReal *uex, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        u0[i]  = user->alpha * (1.0 - x) + user->beta * x;
        uex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    return 0;
}

//STARTFUNCTIONS
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *u,
                                 PetscReal *FF, AppCtx *user) {
    PetscInt   i;
    PetscReal  h = 1.0 / (info->mx-1), x, R;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            FF[i] = u[i] - user->alpha;
        } else if (i == info->mx-1) {
            FF[i] = u[i] - user->beta;
        } else {  // interior location
            if (i == 1) {
                FF[i] = - u[i+1] + 2.0 * u[i] - user->alpha;
            } else if (i == info->mx-2) {
                FF[i] = - user->beta + 2.0 * u[i] - u[i-1];
            } else {
                FF[i] = - u[i+1] + 2.0 * u[i] - u[i-1];
            }
            R = - user->rho * PetscSqrtReal(u[i]);
            x = i * h;
            FF[i] -= h*h * (R + f_source(x));
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
            col[0] = i;
            v[0] = 2.0;
            if (!user->noRinJ) {
                dRdu = - (user->rho / 2.0) / PetscSqrtReal(u[i]);
                v[0] -= h*h * dRdu;
            }
            col[1] = i-1;   v[1] = (i > 1) ? - 1.0 : 0.0;
            col[2] = i+1;   v[2] = (i < info->mx-2) ? - 1.0 : 0.0;
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
//ENDFUNCTIONS
