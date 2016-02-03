static char help[] =
"Solves coupled reaction-diffusion equations (Pearson 1993).  Option prefix -ptn_.\n"
"\n\n";

#include <petsc.h>


typedef struct {
  double u, v;
} Field;

typedef struct {
  DM        da;
  double    L,    // domain side length
            Du,   // diffusion coefficient of first equation
            Dv,   // diffusion coefficient of second equation
            F,    // "dimensionless feed rate" (Pearson 1993)
            k;    // "dimensionless rate constant" (Pearson 1993)
} PtnCtx;

extern PetscErrorCode InitialState(Vec,PtnCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*,double,Field**,Field**,PtnCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PtnCtx         user;
  TS             ts;
  Vec            x;
  double         tf = 10.0;
  int            steps = 10, maxsteps = 200;

  PetscInitialize(&argc,&argv,(char*)0,help);

  // parameter values from pages 21-22 in Hundsdorfer & Verwer (2003)
  user.L  = 2.5;
  user.Du = 8.0e-5;
  user.Dv = 4.0e-5;
  user.F  = 0.024;
  user.k  = 0.06;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "options for patterns", ""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-L","domain side length",
           "pattern.c",user.L,&user.L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Du","diffusion coefficient of first equation",
           "pattern.c",user.Du,&user.Du,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Dv","diffusion coefficient of second equation",
           "pattern.c",user.Dv,&user.Dv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-F","dimensionless feed rate",
           "pattern.c",user.F,&user.F,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-k","dimensionless rate constant",
           "pattern.c",user.k,&user.k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_STAR,
                      -4,-4,PETSC_DECIDE,PETSC_DECIDE,
                      2,  // degrees of freedom
                      1,  // stencil width
                      NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.L, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,0,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,1,"v"); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
                                   (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);

  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,tf); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,tf/steps); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&x); CHKERRQ(ierr);
  ierr = InitialState(x,&user); CHKERRQ(ierr);
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = TSDestroy(&ts); CHKERRQ(ierr);
  ierr = DMDestroy(&user.da); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

// Formulas from page 22 of Hundsdorfer & Verwer (2003)
PetscErrorCode InitialState(Vec x, PtnCtx* user) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  int            i,j;
  double         sx,sy;
  DMDACoor2d     **aC;
  Field          **ax;

  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  ierr = DMDAGetCoordinateArray(user->da,&aC); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,x,&ax); CHKERRQ(ierr);
  for (j = info.ys; j < info.ys+info.ym; j++) {
    for (i = info.xs; i < info.xs+info.xm; i++) {
      if ((aC[j][i].x >= 1.0) && (aC[j][i].x <= 1.5)
              && (aC[j][i].y >= 1.0) && (aC[j][i].y <= 1.5)) {
          sx = sin(4.0 * PETSC_PI * aC[j][i].x);
          sy = sin(4.0 * PETSC_PI * aC[j][i].y);
          ax[j][i].v = 0.5 * sx * sx * sy * sy;
      } else
          ax[j][i].v = 0.0;
      ax[j][i].u = 1.0 - 2.0 * ax[j][i].v;
    }
  }
  ierr = DMDAVecRestoreArray(user->da,x,&ax); CHKERRQ(ierr);
  ierr = DMDARestoreCoordinateArray(user->da,&aC); CHKERRQ(ierr);
  return 0;
}

// in vector form  X_t = G(t,X),  compute G(t,X)
// in scalar form:
//     u_t = G^u(t,u,v) = D_u Laplacian u - u v^2 + F (1 - u)
//     v_t = G^v(t,u,v) = D_v Laplacian v + u v^2 - (F + k) v
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, Field **ax, Field **aG, PtnCtx *user) {
  //PetscErrorCode ierr;
  int            i, j;
  //const double   hx = user->L / (double)(info->mx),  // periodic!
  //               hy = user->L / (double)(info->my);

  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          aG[j][i].u = 0.0;  // FIXME
          aG[j][i].v = 0.0;  // FIXME
      }
  }
  return 0;
}

