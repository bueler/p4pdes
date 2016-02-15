static char help[] =
"Solves coupled reaction-diffusion equations (Pearson 1993).  Option prefix -ptn_.\n"
"\n\n";

//THIS VERSION USES SETIFUNCTION()

// example:
//    ./pattern -ts_monitor_solution draw -da_refine 5 -ptn_tf 2000 -ptn_steps 400 -ts_monitor -snes_converged_reason

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

// in vector form  F(t,X,dot X) = G(t,X),  compute G(t,X)
// in scalar form:
//     G^u(t,u,v) = - u v^2 + F (1 - u)
//     G^v(t,u,v) = + u v^2 - (F + k) v
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, Field **aX,
                                    Field **aG, PtnCtx *user) {
  int            i, j;
  double         uv2;

  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          uv2 = aX[j][i].u * aX[j][i].v * aX[j][i].v;
          aG[j][i].u = - uv2 + user->F * (1.0 - aX[j][i].u);
          aG[j][i].v = + uv2 - (user->F + user->k) * aX[j][i].v;
      }
  }
  return 0;
}


// in vector form  F(t,X,dot X) = G(t,X),  compute F(t,X,dot X)
// in scalar form
//     F^u(t,u,v,u_t,v_t) = u_t - D_u Laplacian u
//     F^v(t,u,v,u_t,v_t) = v_t - D_v Laplacian v
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info, double t, Field **aX,
                                  Field **aXdot, Field **aF, PtnCtx *user) {
  int            i, j;
  const double   h = user->L / (double)(info->mx),
                 Cu = user->Du / (6.0 * h * h),
                 Cv = user->Dv / (6.0 * h * h);
  double         u, v, lapu, lapv;

  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          u = aX[j][i].u;
          v = aX[j][i].v;
          lapu =       aX[j+1][i-1].u + 4.0 * aX[j+1][i].u +     aX[j+1][i+1].u
                 + 4.0 * aX[j][i-1].u -      20.0 * u      + 4.0 * aX[j][i+1].u
                 +     aX[j-1][i-1].u + 4.0 * aX[j-1][i].u +     aX[j-1][i+1].u;
          lapv =       aX[j+1][i-1].v + 4.0 * aX[j+1][i].v +     aX[j+1][i+1].v
                 + 4.0 * aX[j][i-1].v -      20.0 * v      + 4.0 * aX[j][i+1].v
                 +     aX[j-1][i-1].v + 4.0 * aX[j-1][i].v +     aX[j-1][i+1].v;
          aF[j][i].u = aXdot[j][i].u - Cu * lapu;
          aF[j][i].v = aXdot[j][i].v - Cv * lapv;
      }
  }
  return 0;
}



int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PtnCtx         user;
  TS             ts;
  Vec            x;
  DMDALocalInfo  info;
  double         tf = 10.0;
  int            steps = 10;

  PetscInitialize(&argc,&argv,(char*)0,help);

  // parameter values from pages 21-22 in Hundsdorfer & Verwer (2003)
  user.L  = 2.5;
  user.Du = 8.0e-5;
  user.Dv = 4.0e-5;
  user.F  = 0.024;
  user.k  = 0.06;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "options for patterns", ""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-L","square domain side length",
           "pattern.c",user.L,&user.L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Du","diffusion coefficient of first equation",
           "pattern.c",user.Du,&user.Du,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Dv","diffusion coefficient of second equation",
           "pattern.c",user.Dv,&user.Dv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-F","dimensionless feed rate",
           "pattern.c",user.F,&user.F,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-k","dimensionless rate constant",
           "pattern.c",user.k,&user.k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","final time",
           "pattern.c",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps","desired number of time-steps",
           "pattern.c",steps,&steps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,  // for 9-point stencil
                      -3,-3,PETSC_DECIDE,PETSC_DECIDE,
                      2,  // degrees of freedom
                      1,  // stencil width
                      NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "running on %d x %d grid with square cells of side h = %.6f ...\n",
           info.mx,info.my,user.L/(double)(info.mx)); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.L, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,0,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,1,"v"); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
                                   (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(user.da,INSERT_VALUES,
                                 (DMDATSIFunctionLocal)FormIFunctionLocal,&user); CHKERRQ(ierr);

  ierr = TSSetType(ts,TSCN); CHKERRQ(ierr);             // default to Crank-Nicolson
  ierr = TSSetDuration(ts,10*steps,tf); CHKERRQ(ierr);  // allow 10 times requested steps
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

