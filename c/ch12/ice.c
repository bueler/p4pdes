static const char help[] =
"Solves time-dependent nonlinear ice sheet problem in 2D:\n"
"(*)    H_t + div q = m\n"
"where q = (q^x,q^y) is the nonsliding shallow ice approximation (SIA) flux,\n"
"       q = - Gamma H^{n+2} |grad s|^{n-1} grad s\n"
"always subject to the constraint\n"
"       H(t,x,y) >= 0.\n"
"In these equations  H(t,x,y)  is ice thickness,  b(x,y)  is bed elevation,\n"
"s(t,x,y) = H(t,x,y) + b(x,y)  is surface elevation,  and  m(x,y)  is the\n"
"climatic mass balance.  Constants are  n > 1  and  Gamma = 2 A (rho g)^n / (n+2).\n"
"The domain is square,  Omega = [0,L] x [0,L],  with periodic boundary conditions.\n"
"Equation (*) is semi-discretized in space by a Q1 structured-grid FVE method\n"
"(Bueler, 2016); this is method-of-lines.  The resulting ODE in time is\n"
"written in the form\n"
"      F(t,H,H_t) = G(t,H)\n"
"For this ODE we define three callbacks:\n"
"      FormIFunction()   evaluates F(H,H_t) = H_t + div q\n"
"      FormRHSFunction() evaluates G(t,H) = m\n"
"      FormIJacobian()   evaluates (shift) dF/dH_t + dF/dH.\n"
"Options -snes_fd_color works well but is slower than the default Jacobian\n"
"by a factor of two or so.  Requires SNESVI types (i.e.\n"
"-snes_type vinewton{rsls|ssls}) because of constraint.  With current PETSc\n"
"design, explicit TS types do not work.\n\n";

/* try:

./ice                                   # DEFAULT uses analytical jacobian
./ice -snes_fd_color

./ice -ts_view
./ice -da_refine 3                      # only meaningful at this res and higher

./ice -snes_type vinewtonrsls           # DEFAULT
./ice -snes_type vinewtonssls           # slightly more robust?
(other -snes_types like newtonls not allowed because don't support bounds)

./ice -ts_type arkimex                  # DEFAULT
./ice -ts_type beuler                   # robust
./ice -ts_type cn                       # mis-behaves on long time steps
./ice -ts_type cn -ts_theta_adapt       # good
./ice -ts_type theta -ts_theta_adapt    # good
./ice -ts_type bdf -ts_bdf_adapt -ts_bdf_order 2|3|4|5|6  # good

# time-stepping control options
-ts_adapt_type basic                # DEFAULT
-ts_adapt_basic_clip 0.5,1.2        # DEFAULT and recommended:
                                      don't lengthen too much, but allow
                                      significantly shorter, in response to estimate of
                                      local truncation error (default: 0.1,10.0)
-ts_max_snes_failures -1            # recommended: do retry solve
-ts_adapt_scale_solve_failed 0.9    # recommended: try a slightly-easier problem
-ts_max_reject 50                   # recommended?:  keep trying if lte is too big

./ice -ts_monitor -ts_adapt_monitor -ice_dtlimits  # more info on adapt and comparison to explicit

# PC possibilities
-pc_type gamg -pc_gamg_threshold 0.0 -pc_gamg_agg_nsmooths 1  # defaults
-pc_type gamg -pc_gamg_threshold 0.2 -pc_gamg_agg_nsmooths 1  # a little faster?
-pc_type lu
-pc_type ilu
-pc_type asm -sub_pc_type lu
-pc_type asm -sub_pc_type ilu
-pc_type mg
-pc_type mg -pc_mg_levels 4 -mg_levels_ksp_monitor

# shows nontriviality converging ice caps on mountains:
mpiexec -n 2 ./ice -da_refine 5 -ts_monitor_solution draw -snes_converged_reason -ice_tf 10000.0 -ice_dtinit 100.0 -ts_max_snes_failures -1 -ts_adapt_scale_solve_failed 0.9

# start with short time step and it will find good time scale
./ice -snes_converged_reason -da_refine 4 -ice_dtinit 0.1

# recovery from convergence failures works!  (failure here triggered by n=4):
./ice -da_refine 3 -ice_n 4 -ts_max_snes_failures -1 -snes_converged_reason

verif with dome=1, halfar=2:
for TEST in 1 2; do
    for N in 2 3 4 5 6; do
        mpiexec -n 2 ./ice -ice_monitor 0 -ice_verif $TEST -ice_eps 0.0 -ice_dtinit 50.0 -ice_tf 2000.0 -da_refine $N -ts_type beuler
    done
done

actual test B from Bueler et al 2005:
./ice -ice_verif 2 -ice_eps 0 -ice_dtinit 100 -ice_tf 25000 -ice_L 2200e3 -da_refine $N

recommended "new PISM":
mpiexec -n N ./ice -da_refine M \
   -snes_type vinewtonrsls \
   -ts_type arkimex \    #(OR -ts_type bdf -ts_bdf_adapt -ts_bdf_order 4)
   -ts_adapt_type basic -ts_adapt_basic_clip 0.5,1.2 \
   -ts_max_snes_failures -1 -ts_adapt_scale_solve_failed 0.9 \
   -pc_type gamg

run to generate final-time result in file ice_192_50000.dat:
mpiexec -n 4 ./ice -da_refine 6 -pc_type mg -pc_mg_levels 4 -ice_dtlimits -ice_tf 50000 -ice_dtinit 1.0 -ts_max_snes_failures -1 -ts_adapt_scale_solve_failed 0.9 -ice_dump
*/

#include <petsc.h>
#include "icecmb.h"

// context is entirely grid-independent info
typedef struct {
    double    secpera,// number of seconds in a year
              L,      // spatial domain is [0,L] x [0,L]
              tf,     // final time; time domain is [0,tf]
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
              locmaxD,// maximum of diffusivity from last residual evaluation
              dtexplicitsum;// running sum of explicit dt limit
    int       verif;  // 0 = not verification, 1 = dome, 2 = Halfar (1983)
    PetscBool monitor,// use -ice_monitor
              dtlimits,// also monitor time step limits for explicit schemes
              dump;   // dump state (H,b) at final time
    CMBModel  *cmb;// defined in cmbmodel.h
} AppCtx;

#include "iceverif.h"

extern PetscErrorCode SetFromOptionsAppCtx(AppCtx*);
extern PetscErrorCode IceMonitor(TS, int, double, Vec, void*);
extern PetscErrorCode ExplicitLimitsMonitor(TS, int, double, Vec, void*);
extern PetscErrorCode FormBedLocal(DMDALocalInfo*, int, double**, AppCtx*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, double**, AppCtx*);
extern PetscErrorCode FormIJacobianLocal(DMDALocalInfo*, double,
                          double**, double**, double, Mat, Mat, AppCtx *user);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, AppCtx*);

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  TS             ts;
  SNES           snes;   // no need to destroy (owned by TS)
  TSAdapt        adapt;
  Vec            H;
  AppCtx         user;
  CMBModel       cmb;
  DMDALocalInfo  info;
  double         dx,dy,**aH;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SetFromOptionsAppCtx(&user); CHKERRQ(ierr);
  ierr = SetFromOptions_CMBModel(&cmb,user.secpera);
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
     "grid: %d x %d points, spacing dx=%.3f km x dy=%.3f km, dtinit=%.3f a\n",
     user.L/1000.0,user.tf/user.secpera,
     info.mx,info.my,dx/1000.0,dy/1000.0,user.dtinit/user.secpera);

  ierr = DMCreateGlobalVector(da,&H);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"H"); CHKERRQ(ierr);

  // initialize the TS
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX); CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt); CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTBASIC); CHKERRQ(ierr);
  ierr = TSAdaptSetClip(adapt,0.5,1.2); CHKERRQ(ierr);
  ierr = TSSetDM(ts,da); CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,
           (DMDATSIFunctionLocal)FormIFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetIJacobianLocal(da,
           (DMDATSIJacobianLocal)FormIJacobianLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
  if (user.monitor) {
      ierr = TSMonitorSet(ts,IceMonitor,&user,NULL); CHKERRQ(ierr);
  }
  if (user.dtlimits) {
      ierr = TSMonitorSet(ts,ExplicitLimitsMonitor,&user,NULL); CHKERRQ(ierr);
  }

  // configure the SNES to solve NCP/VI at each step
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS); CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds); CHKERRQ(ierr);

  // set time axis
  ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.tf); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,user.dtinit); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
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
      ierr = FormBedLocal(&info,0,aH,&user); CHKERRQ(ierr);  // H(x,y) <- b(x,y)
      ierr = ChopScaleInitialHLocal_CMBModel(&cmb,&info,aH,aH); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(da,H,&aH); CHKERRQ(ierr);

  // solve
  ierr = TSSolve(ts,H); CHKERRQ(ierr);

  // time-stepping summary if -ice_dtlimits
  if (user.dtlimits) {
      int count;
      ierr = TSGetStepNumber(ts,&count); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "average dt %.5f a, average dtexplicit %.5f a\n",
          (user.tf/user.secpera)/(double)count,
          (user.dtexplicitsum/user.secpera)/(double)count); CHKERRQ(ierr);
  }

  // dump state if requested
  if (user.dump) {
      char           filename[1024];
      PetscViewer    viewer;
      Vec            b;
      double         **ab;
      ierr = VecDuplicate(H,&b);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)b,"b"); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da,b,&ab); CHKERRQ(ierr);
      ierr = FormBedLocal(&info,0,ab,&user); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(da,b,&ab); CHKERRQ(ierr);
      ierr = sprintf(filename,"ice_%d_%d.dat",info.mx,(int)(user.tf/user.secpera));
      ierr = PetscPrintf(PETSC_COMM_WORLD,"writing PETSC binary file %s ...\n",filename); CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
      ierr = VecView(b,viewer); CHKERRQ(ierr);
      ierr = VecView(H,viewer); CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
      VecDestroy(&b);
  }

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
          "errors on verif %d: |H-Hexact|_inf = %.3f, |H-Hexact|_average = %.3f\n",
          user.verif,infnorm,onenorm/(double)(info.mx*info.my)); CHKERRQ(ierr);
  }

  // clean up
  VecDestroy(&H);  TSDestroy(&ts);  DMDestroy(&da);
  return PetscFinalize();
}


PetscErrorCode SetFromOptionsAppCtx(AppCtx *user) {
  PetscErrorCode ierr;
  PetscBool      set;

  user->secpera= 31556926.0;  // number of seconds in a year
  user->L      = 1800.0e3;    // m; note  domeL=750.0e3 is radius of verification ice sheet
  user->tf     = 100.0 * user->secpera;  // default to 100 years
  user->dtinit = 10.0 * user->secpera;   // default to 10 year as initial step
  user->g      = 9.81;        // m/s^2
  user->rho_ice= 910.0;       // kg/m^3
  user->n_ice  = 3.0;
  user->A_ice  = 3.1689e-24;  // 1/(Pa^3 s); EISMINT I value
  user->D0     = 1.0;         // m^2 / s
  user->eps    = 0.001;
  user->delta  = 1.0e-4;
  user->lambda = 0.25;
  user->dtexplicitsum = 0.0;
  user->verif  = 0;
  user->monitor = PETSC_TRUE;
  user->dtlimits = PETSC_FALSE;
  user->dump   = PETSC_FALSE;
  user->cmb    = NULL;

  PetscFunctionBeginUser;
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
  ierr = PetscOptionsBool(
      "-dtlimits", "monitor the time-step limits which would apply to an explicit scheme",
      "ice.c",user->dtlimits,&user->dtlimits,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-dump", "save final state (H, b)",
      "ice.c",user->dump,&user->dump,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-eps", "dimensionless regularization for diffusivity D",
      "ice.c",user->eps,&user->eps,NULL);CHKERRQ(ierr);
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


// this basic monitor gives current time, volume, area
PetscErrorCode IceMonitor(TS ts, int step, double time, Vec H, void *ctx) {
    PetscErrorCode ierr;
    AppCtx         *user = (AppCtx*)ctx;
    double         lvol = 0.0, vol, larea = 0.0, area, darea, **aH;
    int            j, k;
    MPI_Comm       com;
    DM             da;
    DMDALocalInfo  info;

    PetscFunctionBeginUser;
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
    PetscFunctionReturn(0);
}

// this monitor reports on max diffusivity, velocity, explicit time-step limits (for comparison)
PetscErrorCode ExplicitLimitsMonitor(TS ts, int step, double time, Vec H, void *ctx) {
    PetscErrorCode ierr;
    AppCtx         *user = (AppCtx*)ctx;
    double         maxD, dd, dtD=PETSC_INFINITY;
    MPI_Comm       com;
    DM             da;
    DMDALocalInfo  info;

    PetscFunctionBeginUser;
    if (time <= 0.0) {
        PetscFunctionReturn(0);
    }
    // globalize maxD
    ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject)(da),&com); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&(user->locmaxD),&maxD,1,MPI_DOUBLE,MPI_MAX,com); CHKERRQ(ierr);
    // compute explicit limits
    if (maxD <= 0.0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    [NO -ice_dtlimits output because maxD is zero]\n"); CHKERRQ(ierr);
    } else {
        ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
        dd = PetscMin(user->L / (double)(info.mx), user->L / (double)(info.my));
        dtD = dd * dd / (4.0*maxD);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"    max D_SIA %.3f m2s-1,  dtD %.3e a\n",
            maxD,dtD/user->secpera); CHKERRQ(ierr);
        user->dtexplicitsum += dtD;
    }
    PetscFunctionReturn(0);
}

PetscErrorCode FormBedLocal(DMDALocalInfo *info, int stencilwidth, double **ab, AppCtx *user) {
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
  PetscFunctionBeginUser;
  // go through owned portion of grid and compute  b(x,y)
  for (k = info->ys-stencilwidth; k < info->ys + info->ym+stencilwidth; k++) {
      y = k * dy;
      for (j = info->xs-stencilwidth; j < info->xs + info->xm+stencilwidth; j++) {
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


/* We factor the SIA flux as
    q = - H^{n+2} sigma(|grad s|) grad s
where sigma is the slope-dependent part
    sigma(z) = Gamma z^{n-1}.
Also
    D = H^{n+2} sigma(|grad s|)
so that q = - D grad s.  */
static double sigma(Grad gH, Grad gb, const AppCtx *user) {
    const double n = user->n_ice;
    if (n > 1.0) {
        const double sx = gH.x + gb.x,
                     sy = gH.y + gb.y,
                     slopesqr = sx * sx + sy * sy + user->delta * user->delta;
        return user->Gamma * PetscPowReal(slopesqr,(n-1.0)/2);
    } else {
        return user->Gamma;
    }
}

/* Regularized derivative of sigma with respect to nodal value H_l:
   d sigma / dl = (d sigma / d gH.x) * (d gH.x / dl) + (d sigma / d gH.y) * (d gH.y / dl),
but
   d sigma / d gH.x = Gamma * (n-1) * |gH + gb|^{n-3.0} * (gH.x + gb.x)
   d sigma / d gH.y = Gamma * (n-1) * |gH + gb|^{n-3.0} * (gH.y + gb.y)
However, power n-3.0 can be negative, which generates NaN in areas where the
surface gradient gH+gb is zero or tiny, so we add delta^2 to |grad s|^2. */
static double DsigmaDl(Grad gH, Grad gb, Grad dgHdl, const AppCtx *user) {
    const double n = user->n_ice;
    if (n > 1.0) {
        const double sx = gH.x + gb.x,
                     sy = gH.y + gb.y,
                     slopesqr = sx * sx + sy * sy + user->delta * user->delta,
                     tmp = user->Gamma * (n-1) * PetscPowReal(slopesqr,(n-3.0)/2);
        return tmp * sx * dgHdl.x + tmp * sy * dgHdl.y;
    } else {
        return 0.0;
    }
}

/* Pseudo-velocity from bed slope:  W = - sigma * grad b. */
static Grad W(double sigma, Grad gb) {
    Grad W;
    W.x = - sigma * gb.x;
    W.y = - sigma * gb.y;
    return W;
}

/* Derivative of pseudo-velocity W with respect to nodal value H_l. */
static Grad DWDl(double dsigmadl, Grad gb) {
    Grad dWdl;
    dWdl.x = - dsigmadl * gb.x;
    dWdl.y = - dsigmadl * gb.y;
    return dWdl;
}


/* DCS = diffusivity from the continuation scheme:
     D(eps) = (1-eps) sigma H^{n+2} + eps D_0
so D(1)=D_0 and D(0)=sigma H^{n+2}. */
static double DCS(double sigma, double H, const AppCtx *user) {
  return (1.0 - user->eps) * sigma * PetscPowReal(PetscAbsReal(H),user->n_ice+2.0)
         + user->eps * user->D0;
}


/* Derivative of diffusivity D = DCS with respect to nodal value H_l.  Since
  D = (1-eps) sigma H^{n+2} + eps D0
it follows that
  d D / dl = (1-eps) [ (d sigma / dl) H^{n+2} + sigma (n+2) H^{n+1} (d H / dl) ]
           = (1-eps) H^{n+1} [ (d sigma / dl) H + sigma (n+2) (d H / dl) ]    */
static double DDCSDl(double sigma, double dsigmadl, double H, double dHdl,
                     const AppCtx *user) {
    const double Hpow = PetscPowReal(PetscAbsReal(H),user->n_ice+1.0);
    return (1.0 - user->eps) * Hpow * ( dsigmadl * H + sigma * (user->n_ice+2.0) * dHdl );
}


/* Flux component from the non-sliding SIA on a general bed. */
PetscErrorCode SIAflux(Grad gH, Grad gb, double H, double Hup, PetscBool xdir,
                       double *D, double *q, const AppCtx *user) {
  const double mysig = sigma(gH,gb,user),
               myD   = DCS(mysig,H,user);
  const Grad   myW   = W(mysig,gb);
  PetscFunctionBeginUser;
  if (D) {
      *D = myD;
  }
  if (xdir && q) {
      *q = - myD * gH.x + myW.x * PetscPowReal(PetscAbsReal(Hup),user->n_ice+2.0);
  } else {
      *q = - myD * gH.y + myW.y * PetscPowReal(PetscAbsReal(Hup),user->n_ice+2.0);
  }
  PetscFunctionReturn(0);
}


static double DSIAfluxDl(Grad gH, Grad gb, Grad dgHdl,
                         double H, double dHdl, double Hup, double dHupdl,
                         PetscBool xdir, const AppCtx *user) {
    const double Huppow   = PetscPowReal(PetscAbsReal(Hup),user->n_ice+1.0),
                 dHuppow  = (user->n_ice+2.0) * Huppow * dHupdl,
                 mysig    = sigma(gH,gb,user),
                 myD      = DCS(mysig,H,user),
                 dsigmadl = DsigmaDl(gH,gb,dgHdl,user),
                 dDdl     = DDCSDl(mysig,dsigmadl,H,dHdl,user);
    const Grad   myW      = W(mysig,gb),
                 dWdl     = DWDl(dsigmadl,gb);
    if (xdir)
        return - dDdl * gH.x - myD * dgHdl.x + dWdl.x * Huppow * Hup + myW.x * dHuppow;
    else
        return - dDdl * gH.y - myD * dgHdl.y + dWdl.y * Huppow * Hup + myW.y * dHuppow;
}


// gradients of weights for Q^1 interpolant
static const double gx[4] = {-1.0,  1.0, 1.0, -1.0},
                    gy[4] = {-1.0, -1.0, 1.0,  1.0};


static double fieldatpt(double xi, double eta, double f[4]) {
  // weights for Q^1 interpolant
  double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
         y[4] = {1.0-eta, 1.0-eta, eta,    eta};
  return   x[0] * y[0] * f[0] + x[1] * y[1] * f[1]
         + x[2] * y[2] * f[2] + x[3] * y[3] * f[3];
}


static double fieldatptArray(int u, int v, double xi, double eta, double **f) {
  double ff[4] = {f[v][u], f[v][u+1], f[v+1][u+1], f[v+1][u]};
  return fieldatpt(xi,eta,ff);
}


static double dfieldatpt(int l, double xi, double eta) {
  const double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
               y[4] = {1.0-eta, 1.0-eta, eta,    eta};
  return x[l] * y[l];
}


static Grad gradfatpt(double xi, double eta, double dx, double dy, double f[4]) {
  Grad gradf;
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


static Grad gradfatptArray(int u, int v, double xi, double eta, double dx, double dy,
                           double **f) {
  double ff[4] = {f[v][u], f[v][u+1], f[v+1][u+1], f[v+1][u]};
  return gradfatpt(xi,eta,dx,dy,ff);
}


static Grad dgradfatpt(int l, double xi, double eta, double dx, double dy) {
  Grad dgradfdl;
  const double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
               y[4] = {1.0-eta, 1.0-eta, eta,    eta},
               gx[4] = {-1.0,  1.0, 1.0, -1.0},
               gy[4] = {-1.0, -1.0, 1.0,  1.0};
  dgradfdl.x = gx[l] *  y[l] / dx;
  dgradfdl.y =  x[l] * gy[l] / dy;
  return dgradfdl;
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
  double          H, Hup, lxup, lyup, **aqquad[4], **ab, DSIA_ckj, qSIA_ckj;
  Grad            gH, gb;
  Vec             qquad[4], b;

  PetscFunctionBeginUser;

#if 0
  // optionally check admissibility
  for (k = info->ys; k < info->ys + info->ym; k++) {
      for (j = info->xs; j < info->xs + info->xm; j++) {
          if (aH[k][j] < 0.0) {
              SETERRQ3(PETSC_COMM_WORLD,1,"ERROR: non-admissible H[k][j] = %.3e < 0.0 detected at j,k = %d,%d ... stopping\n",aH[k][j],j,k);
          }
      }
  }
#endif

  user->locmaxD = 0.0;
  ierr = DMGetLocalVector(info->da, &b); CHKERRQ(ierr);
  if (user->verif > 0) {
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  } else {
      ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
      ierr = FormBedLocal(info,1,ab,user); CHKERRQ(ierr);  // get stencil width
  }
  for (c = 0; c < 4; c++) {
      ierr = DMGetLocalVector(info->da, &(qquad[c])); CHKERRQ(ierr);
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
              ierr = SIAflux(gH,gb,H,Hup,xdire[c],
                             &DSIA_ckj,&qSIA_ckj,user); CHKERRQ(ierr);
              aqquad[c][k][j] = qSIA_ckj;
              if (user->dtlimits) {
                  user->locmaxD = PetscMax(user->locmaxD,DSIA_ckj);
              }
          }
      }
  }

  // loop over nodes, not including ghosts, to get function F(t,H,H') from quadature over
  // s = 0,1,...,7 points on boundary of control volume (rectangle) around node
  for (k=info->ys; k<info->ys+info->ym; k++) {
      for (j=info->xs; j<info->xs+info->xm; j++) {
          FF[k][j] = aHdot[k][j];
          // now add integral over control volume boundary using two
          // quadrature points on each side
          for (s=0; s<8; s++)
              FF[k][j] += coeff[s] * aqquad[ce[s]][k+ke[s]][j+je[s]] / (dx * dy);
      }
  }

  for (c = 0; c < 4; c++) {
      ierr = DMDAVecRestoreArray(info->da,qquad[c],&(aqquad[c])); CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(info->da, &(qquad[c])); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(info->da, &b); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// use j,k for x,y directions in loops and MatSetValuesStencil
typedef struct {
  PetscInt foo,k,j,bar;
} MyStencil;


PetscErrorCode FormIJacobianLocal(DMDALocalInfo *info, double t,
                                  double **aH, double **aHdot, double shift,
                                  Mat J, Mat P, AppCtx *user) {
  PetscErrorCode  ierr;
  const double    dx = user->L / (double)(info->mx),
                  dy = user->L / (double)(info->my);
  const double    coeff[8] = {dy/2, dx/2, dx/2, -dy/2, -dy/2, -dx/2, -dx/2, dy/2};
  const PetscBool upwind = (user->lambda > 0.0);
  const double    upmin = (1.0 - user->lambda) * 0.5,
                  upmax = (1.0 + user->lambda) * 0.5;
  int             j, k, c, l, s, u, v;
  double          H, Hup, **aDqDlquad[16], **ab, val[33], DqSIADl_clkj;
  Grad            gH, gb;
  Vec             DqDlquad[16], b;
  MyStencil       col[33],row;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(P); CHKERRQ(ierr);  // because using ADD_VALUES below

  ierr = DMGetLocalVector(info->da, &b); CHKERRQ(ierr);
  if (user->verif > 0) {
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  } else {
      ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
      ierr = FormBedLocal(info,1,ab,user); CHKERRQ(ierr);  // get stencil width
  }
  for (c = 0; c < 16; c++) {
      ierr = DMGetLocalVector(info->da, &(DqDlquad[c])); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,DqDlquad[c],&(aDqDlquad[c])); CHKERRQ(ierr);
  }

  // loop over locally-owned elements, including ghosts, to get DfluxDl for
  // l=0,1,2,3 at c=0,1,2,3 points in element;  note start at (xs-1,ys-1)
  for (k = info->ys-1; k < info->ys + info->ym; k++) {
      for (j = info->xs-1; j < info->xs + info->xm; j++) {
          for (c=0; c<4; c++) {
              double lxup = locx[c], lyup = locy[c];
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
              } else {
                  Hup = H;
              }
              for (l=0; l<4; l++) {
                  Grad   dgHdl;
                  double dHdl, dHupdl;
                  dgHdl  = dgradfatpt(l,locx[c],locy[c],dx,dy);
                  dHdl   = dfieldatpt(l,locx[c],locy[c]);
                  dHupdl = (upwind) ? dfieldatpt(l,lxup,lyup) : dHdl;
                  DqSIADl_clkj = DSIAfluxDl(gH,gb,dgHdl,H,dHdl,Hup,dHupdl,xdire[c],user);
                  aDqDlquad[4*c+l][k][j] = DqSIADl_clkj;
              }
          }
      }
  }

  // loop over nodes, not including ghosts, to get derivative of residual
  // with respect to nodal value
  for (k=info->ys; k<info->ys+info->ym; k++) {
      row.k = k;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          row.j = j;
          for (s=0; s<8; s++) {
              u = j + je[s];
              v = k + ke[s];
              for (l=0; l<4; l++) {
                  const int djfroml[4] = { 0,  1,  1,  0},
                            dkfroml[4] = { 0,  0,  1,  1};
                  col[4*s+l].j = u + djfroml[l];
                  col[4*s+l].k = v + dkfroml[l];
                  val[4*s+l]   = coeff[s] * aDqDlquad[4*ce[s]+l][v][u] / (dx * dy);
              }
          }
          col[32].j = j;
          col[32].k = k;
          val[32]   = shift;
          ierr = MatSetValuesStencil(P,1,(MatStencil*)&row,
                                     33,(MatStencil*)col,val,ADD_VALUES);CHKERRQ(ierr);
      }
  }

  for (c = 0; c < 16; c++) {
      ierr = DMDAVecRestoreArray(info->da,DqDlquad[c],&(aDqDlquad[c])); CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(info->da, &(DqDlquad[c])); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(info->da, &b); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != P) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  //ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t, double **aH,
                                    double **GG, AppCtx *user) {
  PetscErrorCode  ierr;
  const double    dx = user->L / (double)(info->mx),
                  dy = user->L / (double)(info->my);
  int             j, k;
  Vec             b;
  double          **ab, y, x, m;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(info->da, &b); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = FormBedLocal(info,0,ab,user); CHKERRQ(ierr);  // stencil width NOT needed
  for (k=info->ys; k<info->ys+info->ym; k++) {
      y = k * dy;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          x = j * dx;
          if (user->verif == 1) {
              m = DomeCMB(x,y,user);
          } else if (user->verif == 2) {
              m = 0.0;
          } else {
              m = M_CMBModel(user->cmb,ab[k][j] + aH[k][j]);  // s = b + H is surface elevation
          }
          GG[k][j] = m;
      }
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(info->da, &b); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

