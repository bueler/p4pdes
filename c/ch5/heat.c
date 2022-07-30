static char help[] =
"Solves time-dependent heat equation in 2D using TS.  Option prefix -ht_.\n"
"Equation is  u_t = D_0 laplacian u + f.  Domain is (0,1) x (0,1).\n"
"Boundary conditions are non-homogeneous Neumann in x and periodic in y.\n"
"Energy is conserved (for these particular conditions/source) and an extra\n"
"'monitor' is demonstrated.  Discretization is by centered finite differences.\n"
"Converts the PDE into a system  X_t = G(t,X) (PETSc type 'nonlinear') by\n"
"method of lines.  Uses backward Euler time-stepping by default.\n";

#include <petsc.h>

typedef struct {
  PetscReal D0;    // conductivity
} HeatCtx;

static PetscReal f_source(PetscReal x, PetscReal y) {
    return 3.0 * PetscExpReal(-25.0 * (x-0.6) * (x-0.6))
               * PetscSinReal(2.0*PETSC_PI*y);
}

static PetscReal gamma_neumann(PetscReal y) {
    return PetscSinReal(6.0 * PETSC_PI * y);
}

extern PetscErrorCode Spacings(DMDALocalInfo*, PetscReal*, PetscReal*);
extern PetscErrorCode EnergyMonitor(TS, PetscInt, PetscReal, Vec, void*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal, PetscReal**,
                                           PetscReal**, HeatCtx*);
extern PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo*, PetscReal, PetscReal**,
                                           Mat, Mat, HeatCtx*);

int main(int argc,char **argv) {
  HeatCtx        user;
  TS             ts;
  Vec            u;
  DM             da;
  DMDALocalInfo  info;
  PetscReal      t0, tf;
  PetscBool      monitorenergy = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  user.D0  = 1.0;
  PetscOptionsBegin(PETSC_COMM_WORLD, "ht_", "options for heat", "");
  PetscCall(PetscOptionsReal("-D0","constant thermal diffusivity",
           "heat.c",user.D0,&user.D0,NULL));
  PetscCall(PetscOptionsBool("-monitor","also display total heat energy at each step",
           "heat.c",monitorenergy,&monitorenergy,NULL));
  PetscOptionsEnd();

//STARTDMDASETUP
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR,
      5,4,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.25 grid
      1,1,                            // degrees of freedom, stencil width
      NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&u));
//ENDDMDASETUP

//STARTTSSETUP
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetApplicationContext(ts,&user));
  PetscCall(DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user));
  PetscCall(DMDATSSetRHSJacobianLocal(da,
           (DMDATSRHSJacobianLocal)FormRHSJacobianLocal,&user));
  if (monitorenergy) {
      PetscCall(TSMonitorSet(ts,EnergyMonitor,&user,NULL));
  }
  PetscCall(TSSetType(ts,TSBDF));
  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetMaxTime(ts,0.1));
  PetscCall(TSSetTimeStep(ts,0.001));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
//ENDTSSETUP

  // report on set up
  PetscCall(TSGetTime(ts,&t0));
  PetscCall(TSGetMaxTime(ts,&tf));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
           "solving on %d x %d grid for t0=%g to tf=%g ...\n",
           info.mx,info.my,t0,tf));

  // solve
  PetscCall(VecSet(u,0.0));   // initial condition
  PetscCall(TSSolve(ts,u));

  PetscCall(VecDestroy(&u));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode Spacings(DMDALocalInfo *info, PetscReal *hx, PetscReal *hy) {
    if (hx)  *hx = 1.0 / (PetscReal)(info->mx-1);
    if (hy)  *hy = 1.0 / (PetscReal)(info->my);   // periodic direction
    return 0;
}

//STARTMONITOR
PetscErrorCode EnergyMonitor(TS ts, PetscInt step, PetscReal time, Vec u,
                             void *ctx) {
    HeatCtx        *user = (HeatCtx*)ctx;
    PetscReal      lenergy = 0.0, energy, dt, hx, hy, **au;
    PetscInt       i,j;
    MPI_Comm       com;
    DM             da;
    DMDALocalInfo  info;

    PetscCall(TSGetDM(ts,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(DMDAVecGetArrayRead(da,u,&au));
    for (j = info.ys; j < info.ys + info.ym; j++) {
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if ((i == 0) || (i == info.mx-1))
                lenergy += 0.5 * au[j][i];
            else
                lenergy += au[j][i];
        }
    }
    PetscCall(DMDAVecRestoreArrayRead(da,u,&au));
    PetscCall(Spacings(&info,&hx,&hy));
    lenergy *= hx * hy;
    PetscCall(PetscObjectGetComm((PetscObject)(da),&com));
    PetscCall(MPI_Allreduce(&lenergy,&energy,1,MPIU_REAL,MPIU_SUM,com));
    PetscCall(TSGetTimeStep(ts,&dt));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  energy = %9.2e     nu = %8.4f\n",
                energy,user->D0*dt/(hx*hy)));
    return 0;
}
//ENDMONITOR

//STARTRHSFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info,
                                    PetscReal t, PetscReal **au,
                                    PetscReal **aG, HeatCtx *user) {
  PetscInt   i, j, mx = info->mx;
  PetscReal  hx, hy, x, y, ul, ur, uxx, uyy;

  PetscCall(Spacings(info,&hx,&hy));
  for (j = info->ys; j < info->ys + info->ym; j++) {
      y = hy * j;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          x = hx * i;
          // apply Neumann b.c.s
          ul = (i == 0) ? au[j][i+1] + 2.0 * hx * gamma_neumann(y)
                        : au[j][i-1];
          ur = (i == mx-1) ? au[j][i-1] : au[j][i+1];
          uxx = (ul - 2.0 * au[j][i]+ ur) / (hx*hx);
          // DMDA is periodic in y
          uyy = (au[j-1][i] - 2.0 * au[j][i]+ au[j+1][i]) / (hy*hy);
          aG[j][i] = user->D0 * (uxx + uyy) + f_source(x,y);
      }
  }
  return 0;
}
//ENDRHSFUNCTION

//STARTRHSJACOBIAN
PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo *info,
                                    PetscReal t, PetscReal **au,
                                    Mat J, Mat P, HeatCtx *user) {
    PetscInt         i, j, ncols;
    const PetscReal  D = user->D0;
    PetscReal        hx, hy, hx2, hy2, v[5];
    MatStencil       col[5],row;

    PetscCall(Spacings(info,&hx,&hy));
    hx2 = hx * hx;  hy2 = hy * hy;
    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;  col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            // set up a standard 5-point stencil for the row
            row.i = i;
            col[0].i = i;
            v[0] = - 2.0 * D * (1.0 / hx2 + 1.0 / hy2);
            col[1].j = j-1;  col[1].i = i;    v[1] = D / hy2;
            col[2].j = j+1;  col[2].i = i;    v[2] = D / hy2;
            col[3].j = j;    col[3].i = i-1;  v[3] = D / hx2;
            col[4].j = j;    col[4].i = i+1;  v[4] = D / hx2;
            ncols = 5;
            // if at the boundary, edit the row back to 4 nonzeros
            if (i == 0) {
                ncols = 4;
                col[3].j = j;  col[3].i = i+1;  v[3] = 2.0 * D / hx2;
            } else if (i == info->mx-1) {
                ncols = 4;
                col[3].j = j;  col[3].i = i-1;  v[3] = 2.0 * D / hx2;
            }
            PetscCall(MatSetValuesStencil(P,1,&row,ncols,col,v,INSERT_VALUES));
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
//ENDRHSJACOBIAN
