static const char help[] =
"Solves time-dependent nonlinear ice sheet problem in 2D:\n"
"(*)    H_t + div q = m - div(V H)\n"
"where q = (q^x,q^y) is the nonsliding shallow ice approximation flux,\n"
"       q = - Gamma H^{n+2} |grad s|^{n-1} grad s\n"
"always subject to the constraint\n"
"       H(t,x,y) >= 0.\n"
"In these equations  H(t,x,y)  is ice thickness,  b(x,y)  is bed elevation,\n"
"s(t,x,y) = H(t,x,y) + b(x,y)  is surface elevation,  V(x,y)  is an imposed\n"
"sliding velocity, and  m(x,y),  the climatic mass balance, is the primary\n"
"source term.  Note  n > 1  and  Gamma = 2 A (rho g)^n / (n+2).\n"
"\n"
"The domain is square  [0,L] x [0,L]  with periodic boundary conditions.\n"
"\n"
"Equation (*) is semi-discretized in space by a Q1 structured-grid FVE method\n"
"(Bueler, 2016), so this is method-of-lines.  The resulting ODE in time is\n"
"written in the form\n"
"      F(H,H_t) = G(H)\n"
"and F,G are supplied to PETSc TS as an IFunction and RHSFunction, resp.\n"
"There is no Jacobian, of either side; -snes_fd_color is the default method.\n"
"Requires SNESVI (-snes_type vinewtonrsls|vinewtonssls) because of constraint.\n\n";

// TODO:   1) only V=0 version so far
//         2) implement IJacobian

/* try:

./ice -ts_view
./ice -da_refine 3                      # only meaningful at this res and higher

./ice -snes_type vinewtonrsls           # DEFAULT
./ice -snes_type vinewtonssls           # slightly more robust?
(other -snes_types like newtonls not allowed because don't support bounds)

./ice -ts_type beuler                   # DEFAULT
./ice -ts_type cn                       # mis-behaves on long time steps
./ice -ts_type cn -ts_theta_adapt       # good
./ice -ts_type theta -ts_theta_adapt    # good
./ice -ts_type bdf -ts_bdf_adapt -ts_bdf_order 2|3|4|5|6  # good

# start with short time step and it will find good time scale
./ice -snes_converged_reason -da_refine 4 -ts_type bdf -ts_bdf_adapt -ts_bdf_order 4 -ice_dtinit 0.1

# recovery from convergence failures works!  (failure here triggered by n=4):
./ice -da_refine 3 -ice_n 4 -ts_max_snes_failures -1 -snes_converged_reason

verif with dome=1, halfar=2:
for TEST in 1 2; do
    for N in 2 3 4 5 6; do
        ./ice -ice_monitor 0 -ice_verif $TEST -ice_eps 0.0 -ice_dtinit 50.0 -ice_tf 2000.0 -da_refine $N
    done
done

actual test B from Bueler et al 2005:
./ice -ice_verif 2 -ice_eps 0 -ice_dtinit 100 -ice_tf 25000 -ice_L 2200e3 -da_refine $N

mpiexec -n 2 ./ice -snes_fd_color -da_refine 5 -ts_monitor_solution draw -snes_converged_reason -ice_tf 10000.0 -ice_dtinit 100.0

for MG:

mpiexec -n 4 ./ice -snes_fd_color -da_refine 7 -ts_monitor_solution draw -snes_converged_reason -ice_tf 2.0 -ice_dtinit 1.0 -ksp_converged_reason -pc_type mg -pc_mg_levels 4 -mg_levels_ksp_monitor

for ASM:

mpiexec -n 4 ./ice -snes_fd_color -da_refine 7 -ts_monitor_solution draw -snes_converged_reason -ice_tf 2.0 -ice_dtinit 1.0 -ksp_converged_reason -pc_type asm -sub_pc_type lu

recommended "new PISM":
mpiexec -n N ./ice -da_refine M \
   -snes_type vinewtonrsls \
   -ts_type bdf -ts_bdf_adapt -ts_bdf_order 4 \
   -ts_max_snes_failures -1 \
   -pc_type asm -sub_pc_type lu
*/

#include <petsc.h>
#include "icecmb.h"

// context is entirely grid-independent info
typedef struct {
    double    secpera,// number of seconds in a year
              L,      // spatial domain is [0,L] x [0,L]
              tf,     // time domain is [0,tf]
              dtinit, // user-requested initial time step
              g,      // acceleration of gravity
              rho_ice,// ice density
              n_ice,  // Glen exponent for SIA flux term
              A_ice,  // ice softness
              Gamma,  // coefficient for SIA flux term
              D0,     // representative value of diffusivity (used in regularizing D)
              eps,    // regularization parameter for D
              delta,  // dimensionless regularization for slope in SIA formulas
              lambda, // amount of upwinding; lambda=0 is none and lambda=1 is "full"
              initmagic;// constant, in years, used to multiply CMB for initial H
    int       verif;  // 0 = not verification, 1 = dome, 2 = Halfar (1983)
    PetscBool monitor;// use -ice_monitor
    CMBModel  *cmb;// defined in cmbmodel.h
} AppCtx;

#include "iceverif.h"

extern PetscErrorCode SetFromOptionsAppCtx(AppCtx*);
extern PetscErrorCode IceMonitor(TS, int, double, Vec, void*);
extern PetscErrorCode FormBedLocal(DMDALocalInfo*, double**, AppCtx*);
extern PetscErrorCode ChopScaleCMBInitialHLocal(DMDALocalInfo*, double**, AppCtx*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, double**, AppCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, AppCtx*);

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  TS             ts;
  SNES           snes;   // no need to destroy (owned by TS)
  Vec            H;
  AppCtx         user;
  CMBModel       cmb;
  DMDALocalInfo  info;
  double         dx,dy,**aH;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SetFromOptionsAppCtx(&user); CHKERRQ(ierr);
  ierr = SetFromOptions_CMBModel(&cmb,"ice_cmb_",user.secpera);
  user.cmb = &cmb;

  // this DMDA is the cell-centered grid
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      3,3,PETSC_DECIDE,PETSC_DECIDE,
                      1, 1,        // dof=1, stencilwidth=1
                      NULL,NULL,&da);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(da, &user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, user.L, 0.0, user.L, 0.0,1.0); CHKERRQ(ierr);

  // report on space-time grid
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = user.L / (double)(info.mx);
  dy = user.L / (double)(info.my);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
     "solving on domain [0,L] x [0,L] (L=%.3f km) and time interval [0,tf] (tf=%.3f a)\n"
     "grid is %d x %d points with spacing dx=%.3f km and dy=%.3f km\n",
     user.L/1000.0,user.tf/user.secpera,info.mx,info.my,dx/1000.0,dy/1000.0);

  ierr = DMCreateGlobalVector(da,&H);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"H"); CHKERRQ(ierr);

  // initialize the TS
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetDM(ts,da); CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,
           (DMDATSIFunctionLocal)FormIFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
  if (user.monitor) {
      ierr = TSMonitorSet(ts,IceMonitor,&user,NULL); CHKERRQ(ierr);
  }

  // configure the SNES to solve NCP/VI at each step
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);

  // set time axis defaults
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,user.dtinit); CHKERRQ(ierr);
  ierr = TSSetDuration(ts,100 * (int) ceil(user.tf/user.dtinit),user.tf); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  // set up initial condition on fine grid
  ierr = DMDAVecGetArray(da,H,&aH); CHKERRQ(ierr);
  if (user.verif == 1) {
      ierr = DomeThicknessLocal(&info,aH,&user); CHKERRQ(ierr);
  } else if (user.verif == 2) {
      double t0;
      ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
      ierr = HalfarThicknessLocal(&info,t0,aH,&user); CHKERRQ(ierr);
  } else {
      // fill H according to chop-scale-CMB
      ierr = ChopScaleCMBInitialHLocal(&info,aH,&user); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(da,H,&aH); CHKERRQ(ierr);

  // solve
  ierr = TSSolve(ts,H); CHKERRQ(ierr);

  // compute error in verification case
  if (user.verif > 0) {
      Vec Hexact;
      double infnorm, onenorm;
      ierr = VecDuplicate(H,&Hexact); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da,Hexact,&aH); CHKERRQ(ierr);
      if (user.verif == 1) {
          ierr = DomeThicknessLocal(&info,aH,&user); CHKERRQ(ierr);
      } else if (user.verif == 2) {
          double tf;
          ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
          ierr = HalfarThicknessLocal(&info,tf,aH,&user); CHKERRQ(ierr);
      } else {
          SETERRQ(PETSC_COMM_WORLD,3,"invalid user.verif ... how did I get here?\n");
      }
      ierr = DMDAVecRestoreArray(da,Hexact,&aH); CHKERRQ(ierr);
      ierr = VecAXPY(H,-1.0,Hexact); CHKERRQ(ierr);    // H <- H + (-1.0) Hexact
      VecDestroy(&Hexact);
      ierr = VecNorm(H,NORM_INFINITY,&infnorm); CHKERRQ(ierr);
      ierr = VecNorm(H,NORM_1,&onenorm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "errors: |u-uexact|_inf = %.3f, |u-uexact|_average = %.3f\n",
          infnorm,onenorm/(double)(info.mx*info.my)); CHKERRQ(ierr);
  }

  // clean up
  VecDestroy(&H);
  TSDestroy(&ts);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}


PetscErrorCode SetFromOptionsAppCtx(AppCtx *user) {
  PetscErrorCode ierr;
  PetscBool      set;

  user->secpera= 31556926.0;
  user->L      = 1800.0e3;    // m; note  domeL=750.0e3 is radius of verification ice sheet
  user->tf     = 100.0 * user->secpera;  // default to 100 years
  user->dtinit = 10.0 * user->secpera;   // default to 10 year as initial step
  user->g      = 9.81;       // m/s^2
  user->rho_ice= 910.0;      // kg/m^3
  user->n_ice  = 3.0;
  user->A_ice  = 3.1689e-24; // 1/(Pa^3 s); EISMINT I value
  user->D0     = 1.0;       // m^2 / s
  user->eps    = 0.001;
  user->delta  = 1.0e-4;
  user->lambda = 0.25;
  user->initmagic = 1000.0;  // a
  user->verif  = 0;
  user->monitor = PETSC_TRUE;
  user->cmb    = NULL;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ice_","options to ice","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-A", "set value of ice softness A in units Pa-3 s-1",
      "ice.c",user->A_ice,&user->A_ice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-D0", "representative value of diffusivity (used in regularizing D) in units m2 s-1",
      "ice.c",user->D0,&user->D0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-delta", "dimensionless regularization for slope in SIA formulas",
      "ice.c",user->delta,&user->delta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-dtinit", "initial time step in seconds; input units are years",
      "ice.c",user->dtinit,&user->dtinit,&set);CHKERRQ(ierr);
  if (set)   user->dtinit *= user->secpera;
  ierr = PetscOptionsReal(
      "-eps", "dimensionless regularization for diffusivity D",
      "ice.c",user->eps,&user->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-initmagic", "constant, in years, used to multiply CMB to get initial iterate for thickness",
      "ice.c",user->initmagic,&user->initmagic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-L", "side length of domain in meters",
      "ice.c",user->L,&user->L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-lambda", "amount of upwinding; lambda=0 is none and lambda=1 is full",
      "ice.c",user->lambda,&user->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-monitor", "use the ice monitor which shows ice sheet volume and area",
      "ice.c",user->monitor,&user->monitor,&set);CHKERRQ(ierr);
  if (!set)   user->monitor = PETSC_TRUE;
  ierr = PetscOptionsReal(
      "-n", "value of Glen exponent n",
      "ice.c",user->n_ice,&user->n_ice,NULL);CHKERRQ(ierr);
  if (user->n_ice <= 1.0) {
      SETERRQ1(PETSC_COMM_WORLD,1,
          "ERROR: n = %f not allowed ... n > 1 is required\n",user->n_ice);
  }
  ierr = PetscOptionsReal(
      "-rho", "ice density in units kg m3",
      "ice.c",user->rho_ice,&user->rho_ice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-tf", "final time in seconds; input units are years",
      "ice.c",user->tf,&user->tf,&set);CHKERRQ(ierr);
  if (set)   user->tf *= user->secpera;
  ierr = PetscOptionsInt(
      "-verif","1 = dome exact solution; 2 = halfar exact solution",
      "ice.c",user->verif,&(user->verif),&set);CHKERRQ(ierr);
  if ((set) && ((user->verif < 0) || (user->verif > 2))) {
      SETERRQ1(PETSC_COMM_WORLD,2,
          "ERROR: verif = %d not allowed ... 0 <= verif <= 2 is required\n",
          user->verif);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // derived constant computed after other ice properties are set
  user->Gamma = 2.0 * PetscPowReal(user->rho_ice*user->g,user->n_ice) 
                    * user->A_ice / (user->n_ice+2.0);

  PetscFunctionReturn(0);
}


PetscErrorCode IceMonitor(TS ts, int step, double time, Vec H, void *ctx) {
    PetscErrorCode ierr;
    AppCtx         *user = (AppCtx*)ctx;
    double         lvol = 0.0, vol, larea = 0.0, area, darea, **aH;
    int            j, k;
    MPI_Comm       com;
    DM             da;
    DMDALocalInfo  info;

    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    darea = user->L * user->L / (double)(info.mx * info.my);
    ierr = DMDAVecGetArrayRead(da,H,&aH); CHKERRQ(ierr);
    for (k = info.ys; k < info.ys + info.ym; k++) {
        for (j = info.xs; j < info.xs + info.xm; j++) {
            if (aH[k][j] > 1.0) {  // for volume/area its helpful to not count tinys
                larea += darea;
                lvol += aH[k][j];
            }
        }
    }
    ierr = DMDAVecRestoreArrayRead(da,H,&aH); CHKERRQ(ierr);
    lvol *= darea;
    ierr = PetscObjectGetComm((PetscObject)(da),&com); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&lvol,&vol,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&larea,&area,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "%3d: time %.3f a,  volume %.1f 10^3 km^3,  area %.1f 10^3 km^2\n",
        step,time/user->secpera,vol/1.0e12,area/1.0e9); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormBedLocal(DMDALocalInfo *info, double **ab, AppCtx *user) {
  int          j,k,r,s;
  const double dx = user->L / (double)(info->mx),
               dy = user->L / (double)(info->my),
               Z = PETSC_PI / user->L;
  double       x, y, b;
  // vaguely-random frequencies and coeffs generated by fiddling; see randbed.py
  const int    nc = 4,
               jc[4] = {1, 3, 6, 8},
               kc[4] = {1, 3, 4, 7};
  const double scalec = 750.0,
               C[4][4] = { { 2.00000000,  0.33000000, -0.55020034,  0.54495520},
                           { 0.50000000,  0.45014486,  0.60551833, -0.52250644},
                           { 0.93812068,  0.32638429, -0.24654812,  0.33887052},
                           { 0.17592361, -0.35496741,  0.22694547, -0.05280704} };
  // go through owned portion of grid and compute  b(x,y)
  for (k = info->ys; k < info->ys + info->ym; k++) {
      y = k * dy;
      for (j = info->xs; j < info->xs + info->xm; j++) {
          x = j * dx;
          // b(x,y) is sum of a few sines
          b = 0.0;
          for (r = 0; r < nc; r++) {
              for (s = 0; s < nc; s++) {
                  b += C[r][s] * sin(jc[r] * Z * x) * sin(kc[s] * Z * y);
              }
          }
          ab[k][j] = scalec * b;
      }
  }
  PetscFunctionReturn(0);
}


PetscErrorCode ChopScaleCMBInitialHLocal(DMDALocalInfo *info, double **aH, AppCtx *user) {
  PetscErrorCode  ierr;
  int             j,k;
  double          M;
  ierr = FormBedLocal(info,aH,user); CHKERRQ(ierr);  // H(x,y) <- b(x,y)
  for (k = info->ys; k < info->ys + info->ym; k++) {
      for (j = info->xs; j < info->xs + info->xm; j++) {
          M = M_CMBModel(user->cmb, aH[k][j]);       // M <- CMB(b(x,y))
          aH[k][j] =  (M < 0.0) ? 0.0 : M;           // H(x,y) <- max{CMB(b(x,y)), 0.0}
          aH[k][j] *= user->initmagic * user->secpera;
      }
  }
  PetscFunctionReturn(0);
}

//  for call-back: tell SNESVI (variational inequality) that we want
//    0.0 <= H < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = VecSet(Xl,0.0); CHKERRQ(ierr);
  ierr = VecSet(Xu,PETSC_INFINITY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// value of gradient at a point
typedef struct {
    double x,y;
} Grad;

double getdelta(Grad gH, Grad gb, const AppCtx *user) {
    const double n = user->n_ice;
    if (n > 1.0) {
        const double sx = gH.x + gb.x,
                        sy = gH.y + gb.y,
                        slopesqr = sx * sx + sy * sy + user->delta * user->delta;
        return user->Gamma * PetscPowReal(slopesqr,(n-1.0)/2);
    } else
        return user->Gamma;
}

Grad getW(double delta, Grad gb) {
    Grad W;
    W.x = - delta * gb.x;
    W.y = - delta * gb.y;
    return W;
}

/* DCS = diffusion from the continuation scheme:
   D(eps) = (1-eps) delta H^{n+2} + eps D_0
so   D(1)=D_0 and D(0)=delta H^{n+2}. */
double DCS(double delta, double H, double n, double eps, double D0) {
  return (1.0 - eps) * delta * PetscPowReal(PetscAbsReal(H),n+2.0) + eps * D0;
}

/* ice flux from the non-sliding SIA on a general bed */
double getSIAflux(Grad gH, Grad gb, double H, double Hup,
                  PetscBool xdir, const AppCtx *user) {
  const double n     = user->n_ice,
               delta = getdelta(gH,gb,user),
               myD   = DCS(delta,H,n,user->eps,user->D0);
  const Grad   myW   = getW(delta,gb);
  if (xdir)
      return - myD * gH.x + myW.x * PetscPowReal(PetscAbsReal(Hup),n+2.0);
  else
      return - myD * gH.y + myW.y * PetscPowReal(PetscAbsReal(Hup),n+2.0);
}

/* velocity from sliding model  FIXME returns 0 */
double getslidingvelocity(Grad gb, double H, double Hup,
                          PetscBool xdir, const AppCtx *user) {
  return 0.0;
}

// gradients of weights for Q^1 interpolant
const double gx[4] = {-1.0,  1.0, 1.0, -1.0},
             gy[4] = {-1.0, -1.0, 1.0,  1.0};

double fieldatpt(double xi, double eta, double f[4]) {
  // weights for Q^1 interpolant
  double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
         y[4] = {1.0-eta, 1.0-eta, eta,    eta};
  return   x[0] * y[0] * f[0] + x[1] * y[1] * f[1]
         + x[2] * y[2] * f[2] + x[3] * y[3] * f[3];
}

double fieldatptArray(int u, int v, double xi, double eta, double **f) {
  double ff[4] = {f[v][u], f[v][u+1], f[v+1][u+1], f[v+1][u]};
  return fieldatpt(xi,eta,ff);
}


Grad gradfatpt(double xi, double eta, double dx, double dy, double f[4]) {
  Grad gradf;
  // weights for Q^1 interpolant
  double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
         y[4] = {1.0-eta, 1.0-eta, eta,    eta};
  gradf.x =   gx[0] * y[0] * f[0] + gx[1] * y[1] * f[1]
            + gx[2] * y[2] * f[2] + gx[3] * y[3] * f[3];
  gradf.y =    x[0] *gy[0] * f[0] +  x[1] *gy[1] * f[1]
            +  x[2] *gy[2] * f[2] +  x[3] *gy[3] * f[3];
  gradf.x /= dx;
  gradf.y /= dy;
  return gradf;
}

Grad gradfatptArray(int u, int v, double xi, double eta, double dx, double dy, double **f) {
  double ff[4] = {f[v][u], f[v][u+1], f[v+1][u+1], f[v+1][u]};
  return gradfatpt(xi,eta,dx,dy,ff);
}

// indexing of the 8 quadrature points along the boundary of the control volume in M*
// point s=0,...,7 is in element (j,k) = (j+je[s],k+ke[s])
static const int  je[8] = {0,  0, -1, -1, -1, -1,  0,  0},
                  ke[8] = {0,  0,  0,  0, -1, -1, -1, -1},
                  ce[8] = {0,  3,  1,  0,  2,  1,  3,  2};

// direction of flux at 4 points in each element
static const PetscBool xdire[4] = {PETSC_TRUE, PETSC_FALSE, PETSC_TRUE, PETSC_FALSE};

// local (element-wise) coords of quadrature points for M*
static const double locx[4] = {  0.5, 0.75,  0.5, 0.25},
                    locy[4] = { 0.25,  0.5, 0.75,  0.5};


/* FormIFunctionLocal  =  IFunction call-back by TS using DMDA info.

Evaluates residual FF on local process patch:
   FF_{j,k} = \int_{\partial V_{j,k}} \mathbf{q} \cdot \mathbf{n}
              - m_{j,k} \Delta x \Delta y
where V_{j,k} is the control volume centered at (x_j,y_k).

Regarding indexing locations along the boundary of the control volume where
flux is evaluated, this figure shows four elements and one control volume
centered at (x_j,y_k).  The boundary of the control volume has 8 points,
numbered s=0,...,7:
   -------------------
  |         |         |
  |    ..2..|..1..    |
  |   3:    |    :0   |
k |--------- ---------|
  |   4:    |    :7   |
  |    ..5..|..6..    |
  |         |         |
   -------------------
            j

Regarding flux-component indexing on the element indexed by (j,k) node,
the value  (aqquad[c])[k][j] for c=0,1,2,3 is an x-component at "*" and
a y-component at "%"; note (x_j,y_k) is lower-left corner:
   -------------------
  |         :         |
  |         *2        |
  |    3    :    1    |
  |....%.... ....%....|
  |         :         |
  |         *0        |
  |         :         |
  @-------------------
(j,k)
*/
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info, double t,
                                  double **aH, double **aHdot, double **FF,
                                  AppCtx *user) {
  PetscErrorCode  ierr;
  const double    dx = user->L / (double)(info->mx),
                  dy = user->L / (double)(info->my);
  // coefficients of quadrature evaluations along the boundary of the control volume in M*
  const double    coeff[8] = {dy/2, dx/2, dx/2, -dy/2, -dy/2, -dx/2, -dx/2, dy/2};
  const PetscBool upwind = (user->lambda > 0.0);
  const double    upmin = (1.0 - user->lambda) * 0.5,
                  upmax = (1.0 + user->lambda) * 0.5;
  int             c, j, k, s;
  double          H, Hup, lxup, lyup, **aqquad[4], **ab;
  Grad            gH, gb;
  Vec             qquad[4], b;

  PetscFunctionBeginUser;

  ierr = DMCreateLocalVector(info->da, &b); CHKERRQ(ierr);
  if (user->verif > 0) {
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
  } else {
      ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
      ierr = FormBedLocal(info,ab,user); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
      ierr = DMLocalToLocalBegin(info->da,b,INSERT_VALUES,b); CHKERRQ(ierr);
      ierr = DMLocalToLocalEnd(info->da,b,INSERT_VALUES,b); CHKERRQ(ierr);
  }
  ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  for (c = 0; c < 4; c++) {
      ierr = DMCreateLocalVector(info->da, &(qquad[c])); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,qquad[c],&(aqquad[c])); CHKERRQ(ierr);
  }

  // loop over locally-owned elements, including ghosts, to get fluxes q at
  // c = 0,1,2,3 points in element;  note start at (xs-1,ys-1)
  for (k = info->ys-1; k < info->ys + info->ym; k++) {
      for (j = info->xs-1; j < info->xs + info->xm; j++) {
          for (c=0; c<4; c++) {
              H  = fieldatptArray(j,k,locx[c],locy[c],aH);
              gH = gradfatptArray(j,k,locx[c],locy[c],dx,dy,aH);
              gb = gradfatptArray(j,k,locx[c],locy[c],dx,dy,ab);
              if (upwind) {
                  if (xdire[c] == PETSC_TRUE) {
                      lxup = (gb.x <= 0.0) ? upmin : upmax;
                      lyup = locy[c];
                  } else {
                      lxup = locx[c];
                      lyup = (gb.y <= 0.0) ? upmin : upmax;
                  }
                  Hup = fieldatptArray(j,k,lxup,lyup,aH);
              } else
                  Hup = H;
              aqquad[c][k][j] = getSIAflux(gH,gb,H,Hup,xdire[c],user);
          }
      }
  }

  // loop over nodes, not including ghosts, to get function F(t,H,H') from quadature over
  // s = 0,1,...,7 points on boundary of control volume (rectangle) around node
  for (k=info->ys; k<info->ys+info->ym; k++) {
      for (j=info->xs; j<info->xs+info->xm; j++) {
          FF[k][j] = aHdot[k][j] * dx * dy;
          // add the integral over the control volume boundary using two
          // quadrature points on each of the four sides of the
          // rectangular control volume
          for (s=0; s<8; s++)
              FF[k][j] += coeff[s] * aqquad[ce[s]][k+ke[s]][j+je[s]];
      }
  }

  for (c = 0; c < 4; c++) {
      ierr = DMDAVecRestoreArray(info->da,qquad[c],&(aqquad[c])); CHKERRQ(ierr);
      ierr = VecDestroy(&(qquad[c])); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, double **aH,
                                    double **GG, AppCtx *user) {
  PetscErrorCode  ierr;
  const double    dx = user->L / (double)(info->mx),
                  dy = user->L / (double)(info->my);
  // coefficients of quadrature evaluations along the boundary of the control volume in M*
  const double    coeff[8] = {dy/2, dx/2, dx/2, -dy/2, -dy/2, -dx/2, -dx/2, dy/2};
  int             j, k, s, c;
  Vec             b, VH[4];
  Grad            gb;
  double          **ab, **aVH[4], y, x, m, H;

  PetscFunctionBeginUser;

  for (c = 0; c < 4; c++) {
      ierr = DMCreateLocalVector(info->da, &(VH[c])); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,VH[c],&(aVH[c])); CHKERRQ(ierr);
  }

  // loop over locally-owned elements, including ghosts, to get fluxes VH at
  // c = 0,1,2,3 points in element;  note start at (xs-1,ys-1)
  for (k = info->ys-1; k < info->ys + info->ym; k++) {
      for (j = info->xs-1; j < info->xs + info->xm; j++) {
          for (c=0; c<4; c++) {
              H  = fieldatptArray(j,k,locx[c],locy[c],aH);
              // FIXME to use bed, need to get stencil width on b as in FormFunctionLocal()
              //gb = gradfatptArray(j,k,locx[c],locy[c],dx,dy,ab);
              gb.x = 0.0;
              gb.y = 0.0;
              // FIXME this form does not do any upwinding:
              aVH[c][k][j] = getslidingvelocity(gb,H,H,xdire[c],user) * H;
          }
      }
  }

  ierr = DMCreateLocalVector(info->da, &b); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = FormBedLocal(info,ab,user); CHKERRQ(ierr);
  for (k=info->ys; k<info->ys+info->ym; k++) {
      y = k * dy;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          x = j * dx;
          if (user->verif == 1) {
              m = DomeCMB(x,y,user);
          } else if (user->verif == 2) {
              m = 0.0;
          } else {
              m = M_CMBModel(user->cmb,ab[k][j] + aH[k][j]);
          }          
          GG[k][j] = m * dx * dy;

          // add the integral over the control volume boundary using two
          // quadrature points on each of the four sides of the
          // rectangular control volume
          for (s=0; s<8; s++)
              GG[k][j] += coeff[s] * aVH[ce[s]][k+ke[s]][j+je[s]];
      }
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);

  for (c = 0; c < 4; c++) {
      ierr = DMDAVecRestoreArray(info->da,VH[c],&(aVH[c])); CHKERRQ(ierr);
      ierr = VecDestroy(&(VH[c])); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

