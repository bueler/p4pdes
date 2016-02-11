static char help[] =
"Solves time-dependent heat equation in 2D using TS.  Option prefix -heat_.\n"
"Equation is  u_t = k laplacian u + f.  Boundary conditions are\n"
"non-homogeneous Neumann in x and periodic in y.  Discretization by\n"
"finite differences.\n\n";

#include <petsc.h>

typedef struct {
  DM     da;
  Vec    f;
  double k,    // conductivity
         Lx,   // domain length in x:  [0,Lx]
         Ly;   // domain length in y:  [0,Ly]
} HeatCtx;


PetscErrorCode InitialState(Vec u, HeatCtx* user) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  int            i,j;
  double         **au;

  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,u,&au); CHKERRQ(ierr);
  for (j = info.ys; j < info.ys+info.ym; j++) {
    for (i = info.xs; i < info.xs+info.xm; i++) {
      au[j][i] = 0.0;  // FIXME function of x,y
    }
  }
  ierr = DMDAVecRestoreArray(user->da,u,&au); CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, double **au,
                                    double **aG, HeatCtx *user) {
  PetscErrorCode ierr;
  int            i, j;
  const double   hx = user->Lx / (double)(info->mx-1),
                 hy = user->Ly / (double)(info->my),   // periodic direction
                 hx2 = hx * hx,  hy2 = hy * hy;
  double         uxx, uyy, **af;

  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          uxx = (au[j][i-1] - 2.0 * au[j][i]+ au[j][i+1]) / hx2;
          uyy = (au[j-1][i] - 2.0 * au[j][i]+ au[j+1][i]) / hy2;
          aG[j][i] = user->k * (uxx + uyy) + 0.0 * af[j][i];
          // FIXME need boundary conditions
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  HeatCtx        user;
  TS             ts;
  Vec            u, uexact;
  DMDALocalInfo  info;
  double         tf = 10.0;
  int            steps = 10;

  PetscInitialize(&argc,&argv,(char*)0,help);

  user.k  = 1.0;
  user.Lx = 1.0;
  user.Ly = 1.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "heat_", "options for heat", ""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-k","dimensionless rate constant",
           "heat.c",user.k,&user.k,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tf","final time",
           "heat.c",tf,&tf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-steps","desired number of time-steps",
           "heat.c",steps,&steps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_STAR,
                      -3,-3,PETSC_DECIDE,PETSC_DECIDE,
                      1,  // degrees of freedom
                      1,  // stencil width
                      NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
           "running on %d x %d grid ...\n",
           info.mx,info.my); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
                                   (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);

  ierr = TSSetType(ts,TSCN); CHKERRQ(ierr);             // default to Crank-Nicolson
  ierr = TSSetDuration(ts,10*steps,tf); CHKERRQ(ierr);  // allow 10 times requested steps
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,tf/steps); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.f)); CHKERRQ(ierr);
  ierr = InitialState(u,&user); CHKERRQ(ierr);
  ierr = TSSolve(ts,u);CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  TSDestroy(&ts);  DMDestroy(&user.da);
  PetscFinalize();
  return 0;
}

