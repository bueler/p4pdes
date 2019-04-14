static char help[] =
"1D reaction-diffusion problem with DMDA and SNES.  Option prefix -rct_.\n\n";

#include <petsc.h>

typedef struct {
    double    rho, M, alpha, beta;
    PetscBool noRinJ;
} AppCtx;

extern double f_source(double);
extern PetscErrorCode InitialAndExact(DMDALocalInfo*, double*, double*, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double*, double*, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, double*, Mat, Mat, AppCtx*);

//STARTMAIN
int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM            da;
  SNES          snes;
  AppCtx        user;
  Vec           u, uexact;
  double        errnorm, *au, *auex;
  DMDALocalInfo info;

  PetscInitialize(&argc,&args,NULL,help);
  user.rho   = 10.0;
  user.M     = PetscSqr(user.rho / 12.0);
  user.alpha = user.M;
  user.beta  = 16.0 * user.M;
  user.noRinJ = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"rct_","options for reaction",""); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-noRinJ","do not include R(u) term in Jacobian",
      "reaction.c",user.noRinJ,&(user.noRinJ),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,u,&au); CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,uexact,&auex); CHKERRQ(ierr);
  ierr = InitialAndExact(&info,au,auex,&user); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,u,&au); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,uexact,&auex); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "on %d point grid:  |u-u_exact|_inf = %g\n",info.mx,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  return PetscFinalize();
}
//ENDMAIN

double f_source(double x) {
    return 0.0;
}

//STARTFUNCTIONS
PetscErrorCode InitialAndExact(DMDALocalInfo *info, double *u0,
                               double *uex, AppCtx *user) {
    int    i;
    double h = 1.0 / (info->mx-1), x;
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = h * i;
        u0[i]  = user->alpha * (1.0 - x) + user->beta * x;
        uex[i] = user->M * PetscPowReal(x + 1.0,4.0);
    }
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double *u,
                                 double *FF, AppCtx *user) {
    int          i;
    const double h = 1.0 / (info->mx-1);
    double       x, R;
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

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, double *u,
                                 Mat J, Mat P, AppCtx *user) {
    PetscErrorCode ierr;
    int    i, col[3];
    double h = 1.0 / (info->mx-1), dRdu, v[3];
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if ((i == 0) | (i == info->mx-1)) {
            v[0] = 1.0;
            ierr = MatSetValues(P,1,&i,1,&i,v,INSERT_VALUES); CHKERRQ(ierr);
        } else {
            col[0] = i;  v[0] = 2.0;
            if (!user->noRinJ) {
                dRdu = - (user->rho / 2.0) / PetscSqrtReal(u[i]);
                v[0] -= h*h * dRdu;
            }
            col[1] = i-1;   v[1] = (i > 1) ? - 1.0 : 0.0;
            col[2] = i+1;   v[2] = (i < info->mx-2) ? - 1.0 : 0.0;
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
//ENDFUNCTIONS

