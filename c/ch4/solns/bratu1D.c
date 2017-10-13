static char help[] = "Solves the Liouville-Bratu reaction-diffusion problem in 1D.  Option prefix lb_.\n"
"Solves\n"
"    - u'' - lambda e^u = 0\n"
"on [0,1] subject to homogeneous Dirichlet boundary conditions.  Optionally\n"
"uses manufactured solution to problem with f(x) on right-hand-side.\n\n";

#include <petsc.h>

typedef struct {
    PetscBool manufactured;
    double    lambda;
} AppCtx;

extern PetscErrorCode Exact(DM, DMDALocalInfo*, Vec, AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double*,
                                 double*, AppCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, double*,
                                 Mat, Mat, AppCtx*);

int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM                  da;
  SNES                snes;
  AppCtx              user;
  Vec                 u, uexact;
  double              unorm, errnorm;
  DMDALocalInfo       info;

  PetscInitialize(&argc,&args,(char*)0,help);
  user.lambda = 1.0;
  user.manufactured = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,
                           "lb_","options for bratu1D",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lambda","coefficient of nonlinear zeroth-order term",
                          "bratu1D.c",user.lambda,&(user.lambda),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-manu","if set, use a manufactured solution",
                          "bratu1D.c",user.manufactured,&(user.manufactured),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,9,1,1,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  if (user.manufactured) {
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

  VecDestroy(&u);  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

PetscErrorCode Exact(DM da, DMDALocalInfo *info, Vec uex, AppCtx *user) {
    PetscErrorCode ierr;
    int    i;
    double h = 1.0 / (info->mx-1), x, *auex;
    ierr = DMDAVecGetArray(da,uex,&auex); CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = i * h;
        auex[i] = sin(PETSC_PI * x);
    }
    ierr = DMDAVecRestoreArray(da,uex,&auex); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double *u,
                                 double *f, AppCtx *user) {
    int    i;
    double h = 1.0 / (info->mx-1), x, R, compf, uex;
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

