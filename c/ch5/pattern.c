static char help[] =
"Coupled reaction-diffusion equations (Pearson 1993).  Option prefix -ptn_.\n"
"Demonstrates form  F(t,X,dot X) = G(t,X)  where F() is IFunction and G() is\n"
"RHSFunction().  Implements IJacobian().  Defaults to ARKIMEX (adaptive\n"
"Runge-Kutta implicit-explicit) type of TS.\n\n";

// modest refinement and a bit of feedback:
//    ./pattern -ts_monitor -ts_final_time 100 -da_refine 5 -snes_converged_reason

// show solution graphically:
//    ./pattern -ts_monitor_solution draw -da_refine 5 -ts_final_time 2000 -ts_dt 5 -ts_monitor -snes_converged_reason
// example works with -ts_fd_color:

// suggests Jacobian is correct (also with -snes_test_display):
//    ./pattern -ptn_L 0.5 -ts_final_time 1 -ts_dt 1 -ts_monitor -snes_converged_reason -snes_type test

// compare runs with
// -dm_mat_type aij
// -dm_mat_type baij
// -dm_mat_type sbaij -ksp_type cg -pc_type icc  # 20% speed up?

#include <petsc.h>

//FIELDCTX
typedef struct {
  double u, v;
} Field;

typedef struct {
  DM        da;
  double    L,    // domain side length
            Du,   // diffusion coefficient of first equation
            Dv,   // diffusion coefficient of second equation
            phi,  // = "dimensionless feed rate" F in (Pearson 1993)
            kappa;// = "dimensionless rate constant" k in (Pearson 1993)
} PtnCtx;
//ENDFIELDCTX

// Formulas from page 22 of Hundsdorfer & Verwer (2003).  Interpretation here is
// to always generate 0.5 x 0.5 non-trivial patch in (0,L) x (0,L) domain.
PetscErrorCode InitialState(Vec x, PtnCtx* user) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  int            i,j;
  double         sx,sy;
  const double   ledge = (user->L - 0.5) / 2.0, // nontrivial initial values on
                 redge = user->L - ledge;       //   ledge < x,y < redge
  DMDACoor2d     **aC;
  Field          **ax;

  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  ierr = DMDAGetCoordinateArray(user->da,&aC); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,x,&ax); CHKERRQ(ierr);
  for (j = info.ys; j < info.ys+info.ym; j++) {
    for (i = info.xs; i < info.xs+info.xm; i++) {
      if ((aC[j][i].x >= ledge) && (aC[j][i].x <= redge)
              && (aC[j][i].y >= ledge) && (aC[j][i].y <= redge)) {
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

// in system form  F(t,X,dot X) = G(t,X),  compute G():
//     G^u(t,u,v) = - u v^2 + phi (1 - u)
//     G^v(t,u,v) = + u v^2 - (phi + kappa) v
//RHSFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, Field **aX,
                                    Field **aG, PtnCtx *user) {
  int            i, j;
  double         uv2;

  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          uv2 = aX[j][i].u * aX[j][i].v * aX[j][i].v;
          aG[j][i].u = - uv2 + user->phi * (1.0 - aX[j][i].u);
          aG[j][i].v = + uv2 - (user->phi + user->kappa) * aX[j][i].v;
      }
  }
  return 0;
}
//ENDRHSFUNCTION


// in system form  F(t,X,dot X) = G(t,X),  compute F():
//     F^u(t,u,v,u_t,v_t) = u_t - D_u Laplacian u
//     F^v(t,u,v,u_t,v_t) = v_t - D_v Laplacian v
//IFUNCTION
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
          lapu =     aX[j+1][i-1].u + 4.0*aX[j+1][i].u +   aX[j+1][i+1].u
                 + 4.0*aX[j][i-1].u -    20.0*u        + 4.0*aX[j][i+1].u
                 +   aX[j-1][i-1].u + 4.0*aX[j-1][i].u +   aX[j-1][i+1].u;
          lapv =     aX[j+1][i-1].v + 4.0*aX[j+1][i].v +   aX[j+1][i+1].v
                 + 4.0*aX[j][i-1].v -    20.0*v        + 4.0*aX[j][i+1].v
                 +   aX[j-1][i-1].v + 4.0*aX[j-1][i].v +   aX[j-1][i+1].v;
          aF[j][i].u = aXdot[j][i].u - Cu * lapu;
          aF[j][i].v = aXdot[j][i].v - Cv * lapv;
      }
  }
  return 0;
}
//ENDIFUNCTION


// in system form  F(t,X,dot X) = G(t,X),  compute combined/shifted
// Jacobian of F():
//     J = (shift) dF/d(dot X) + dF/dX
//IJACOBIAN
PetscErrorCode FormIJacobianLocal(DMDALocalInfo *info, double t, Field **aX,
                                  Field **aXdot, double shift, Mat J, Mat P,
                                  PtnCtx *user) {
    PetscErrorCode ierr;
    int            i, j, s, c;
    const double   h = user->L / (double)(info->mx),
                   Cu = user->Du / (6.0 * h * h),
                   Cv = user->Dv / (6.0 * h * h);
    double         val[9], CC;
    MatStencil     col[9], row;

    //ierr = PetscPrintf(PETSC_COMM_WORLD,"IJacobian() called at t=%g\n",t); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        row.j = j;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            row.i = i;
            for (c = 0; c < 2; c++) { // u,v equations are c=0,1
                row.c = c;
                CC = (c == 0) ? Cu : Cv;
                for (s = 0; s < 9; s++)
                    col[s].c = c;
                col[0].i = i;   col[0].j = j;    val[0] = shift + 20.0 * CC;
                col[1].i = i-1; col[1].j = j;    val[1] = - 4.0 * CC;
                col[2].i = i+1; col[2].j = j;    val[2] = - 4.0 * CC;
                col[3].i = i;   col[3].j = j-1;  val[3] = - 4.0 * CC;
                col[4].i = i;   col[4].j = j+1;  val[4] = - 4.0 * CC;
                col[5].i = i-1; col[5].j = j-1;  val[5] = - CC;
                col[6].i = i-1; col[6].j = j+1;  val[6] = - CC;
                col[7].i = i+1; col[7].j = j-1;  val[7] = - CC;
                col[8].i = i+1; col[8].j = j+1;  val[8] = - CC;
                ierr = MatSetValuesStencil(P,1,&row,9,col,val,INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }

    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}
//ENDIJACOBIAN


int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PtnCtx         user;
  TS             ts;
  Vec            x;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,(char*)0,help);

  // parameter values from pages 21-22 in Hundsdorfer & Verwer (2003)
  user.L      = 2.5;
  user.Du     = 8.0e-5;
  user.Dv     = 4.0e-5;
  user.phi    = 0.024;
  user.kappa  = 0.06;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "ptn_", "options for patterns", ""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-L","square domain side length; recommend L >= 0.5",
           "pattern.c",user.L,&user.L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Du","diffusion coefficient of first equation",
           "pattern.c",user.Du,&user.Du,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Dv","diffusion coefficient of second equation",
           "pattern.c",user.Dv,&user.Dv,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-phi","dimensionless feed rate (=F in (Pearson, 1993))",
           "pattern.c",user.phi,&user.phi,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-kappa","dimensionless rate constant (=k in (Pearson, 1993))",
           "pattern.c",user.kappa,&user.kappa,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,  // for 9-point stencil
                      -4,-4,PETSC_DECIDE,PETSC_DECIDE,
                      2,  // degrees of freedom
                      1,  // stencil width
                      NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  if (info.mx != info.my) {
      SETERRQ(PETSC_COMM_WORLD,1,"pattern.c requires mx == my");
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "running on %d x %d grid with square cells of side h = %.6f ...\n",
           info.mx,info.my,user.L/(double)(info.mx)); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.L, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,0,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,1,"v"); CHKERRQ(ierr);

//TSSETUP
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(user.da,INSERT_VALUES,
           (DMDATSIFunctionLocal)FormIFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetIJacobianLocal(user.da,
           (DMDATSIJacobianLocal)FormIJacobianLocal,&user); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,5.0); CHKERRQ(ierr);  // t_0 = 0.0, dt = 5.0
  ierr = TSSetDuration(ts,1000000,200.0); CHKERRQ(ierr);   // t_f = 200
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
//ENDTSSETUP

  ierr = DMCreateGlobalVector(user.da,&x); CHKERRQ(ierr);
  ierr = InitialState(x,&user); CHKERRQ(ierr);
  ierr = TSSolve(ts,x); CHKERRQ(ierr);

  VecDestroy(&x);  TSDestroy(&ts);  DMDestroy(&user.da);
  PetscFinalize();
  return 0;
}

