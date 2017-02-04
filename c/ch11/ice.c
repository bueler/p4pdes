static const char help[] =
"Solves time-dependent nonlinear ice sheet problem in 2D:\n"
"(*)    H_t + div (q^x,q^y) = m - div(V H)\n"
"where q is the nonsliding shallow ice approximation flux,\n"
"      (q^x,q^y) = - Gamma H^{n+2} |grad s|^{n-1} grad s.\n"
"In these equations  H(x,y)  is ice thickness,  b(x,y)  isbed elevation,\n"
"s(x,y) = H(x,y) + b(x,y)  is surface elevation, and V(x,y) is an imposed\n"
"sliding velocity.  Note  n > 1  and  Gamma = 2 A (rho g)^n / (n+2).\n"
"Equation (*) is semi-discretized in space and then treated as an ODE in time\n"
"in the form\n"
"      F(H,H_t) = G(H)\n"
"and F,G are supplied to PETSc TS as an IFunction and RHSFunction, resp.\n"
"Structured-grid on a square domain  [0,L] x [0,L]  with periodic boundary.\n"
"Computed by Q1 FVE method (Bueler, 2016) with FD evaluation of Jacobian.\n"
"Uses SNESVI because of constraint  H(x,y) >= 0.\n\n";

#include <petsc.h>

typedef struct {
  PetscReal ela,   // equilibrium line altitude (m)
            zgrad; // vertical derivative (gradient) of CMB (s^-1)
} CMBModel;

typedef struct {
  // grid independent data:
  PetscReal L,      // spatial domain is [0,L] x [0,L]
            tf,     // time domain is [0,tf]
            secpera,// number of seconds in a year
            g,      // acceleration of gravity
            rho_ice,// ice density
            n_ice,  // Glen exponent for SIA flux term
            A_ice, // ice softness
            Gamma,  // coefficient for SIA flux term
            eps,    // regularization parameter for D
            delta,  // dimensionless regularization for slope in SIA formulas
            lambda, // amount of upwinding; lambda=0 is none and lambda=1 is "full"
            initmagic;// constant, in years, used to multiply CMB fo initial H
  CMBModel  *cmb;
  // describe the fine grid:
  DM        da;
  Vec       Hexact, // the exact thickness (valid in verification case)
            Hinit;  // initial state
  double    dx, dy; // grid spacings
} AppCtx;


extern PetscErrorCode SetFromOptionsAppCtx(AppCtx*);
extern PetscErrorCode SetFromOptionsCMBModel(CMBModel*, const char*, double);
extern PetscErrorCode M_CMBModel(CMBModel*, double, double*);
extern PetscErrorCode dMds_CMBModel(CMBModel*, double, double*);
extern PetscErrorCode ChopScaleCMBforInitialH(Vec,AppCtx*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, double**, AppCtx*);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, double,
                          double**, double**, AppCtx*);

int main(int argc,char **argv) {
  PetscErrorCode      ierr;
  TS                  ts;
  SNES                snes;
  Vec                 H;
  AppCtx              user;
  CMBModel            cmb;
  DMDALocalInfo       info;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SetFromOptionsAppCtx(&user); CHKERRQ(ierr);
  ierr = SetFromOptionsCMBModel(&cmb,"cmb_",user.secpera);
  user.cmb = &cmb;

  // this DMDA is used for scalar fields on nodes; cell-centered grid
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      18,18,PETSC_DECIDE,PETSC_DECIDE,
                      1, 1,        // dof=1, stencilwidth=1
                      NULL,NULL,&user.da);
  ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
  ierr = DMSetUp(user.da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(user.da, &user);CHKERRQ(ierr);

  // compute grid spacing
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  user.dx = user.L / (PetscReal)(info.mx);
  user.dy = user.L / (PetscReal)(info.my);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.L, 0.0, user.L, 0.0,1.0); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "solving on [0,L] x [0,L] with  L=%.3f km;\n"
             "fine grid is  %d x %d  points with spacing  dx = %.6f km  and  dy = %.6f km ...\n",
             user.L/1000.0,info.mx,info.my,user.dx/1000.0,user.dy/1000.0);

  ierr = DMCreateGlobalVector(user.da,&H);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"thickness solution H"); CHKERRQ(ierr);

  // Hexact is valid only in verification case
  ierr = VecDuplicate(H,&user.Hexact); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(user.Hexact),"exact/observed thickness H"); CHKERRQ(ierr);

  // fill user.Hinitial according to chop-scale-CMB
  ierr = VecDuplicate(H,&user.Hinit); CHKERRQ(ierr);
  ierr = ChopScaleCMBforInitialH(user.Hinit,&user); CHKERRQ(ierr);

  // initialize the TS
  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.da); CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(user.da,INSERT_VALUES,
           (DMDATSIFunctionLocal)FormIFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDATSSetRHSFunctionLocal(user.da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
  ierr = TSGetSNES(ts,&snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);

  // FIXME finalize setting of time axis
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.01); CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000000,0.1); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = VecCopy(user.Hinit,H); CHKERRQ(ierr);
  ierr = TSSolve(ts,H); CHKERRQ(ierr);

  // clean up
  VecDestroy(&H);  VecDestroy(&user.Hinit);  VecDestroy(&user.Hexact);
  TSDestroy(&ts);  DMDestroy(&user.da);
  PetscFinalize();
  return 0;
}


PetscErrorCode SetFromOptionsAppCtx(AppCtx *user) {
  PetscErrorCode ierr;

  user->L  = 3.0;
  user->n_ice  = 3.0;
  user->g      = 9.81;       // m/s^2
  user->rho_ice= 910.0;      // kg/m^3
  user->secpera= 31556926.0;
  user->A_ice  = 1.0e-16/user->secpera; // = 3.17e-24  1/(Pa^3 s); EISMINT I value
  user->initmagic = 1000.0;  // a
  user->delta  = 1.0e-4;
  user->lambda = 0.25;  // amount of upwinding; some trial-and-error with bedstep soln; 0.1 gives some Newton convergence difficulties on refined grid (=125m); earlier M* used 0.5
  user->cmb = NULL;
#define domeL  750.0e3 // radius of exact ice sheet
  user->L = 900.0e3;    // m

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ice_","options to ice","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-A", "set value of ice softness A in units Pa-3 s-1",
      "ice.c",user->A_ice,&user->A_ice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-delta", "dimensionless regularization for slope in SIA formulas",
      "ice.c",user->delta,&user->delta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-initmagic", "constant, in years, used to multiply CMB to get initial iterate for thickness",
      "ice.c",user->initmagic,&user->initmagic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-lambda", "amount of upwinding; lambda=0 is none and lambda=1 is full",
      "ice.c",user->lambda,&user->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-n", "value of Glen exponent n",
      "ice.c",user->n_ice,&user->n_ice,NULL);CHKERRQ(ierr);
  if (user->n_ice <= 1.0) {
      SETERRQ1(PETSC_COMM_WORLD,11,"ERROR: n = %f not allowed ... n > 1 is required\n",user->n_ice); }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // derived constant computed after n,A get set
  user->Gamma = 2.0 * PetscPowReal(user->rho_ice*user->g,user->n_ice) * user->A_ice / (user->n_ice+2.0);

  PetscFunctionReturn(0);
}

PetscErrorCode SetFromOptionsCMBModel(CMBModel *cmb, const char *optprefix, PetscReal secpera) {
  PetscErrorCode ierr;
  cmb->ela   = 2000.0; // m
  cmb->zgrad = 0.001;  // a^-1
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,optprefix,
            "options to climatic mass balance (CMB) model, if used","");CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-ela", "equilibrium line altitude, in m",
      "cmbmodel.c",cmb->ela,&cmb->ela,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-zgrad", "vertical derivative (gradient) of CMB, in a^-1",
      "cmbmodel.c",cmb->zgrad,&cmb->zgrad,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  cmb->zgrad /= secpera;
  PetscFunctionReturn(0);
}


PetscErrorCode M_CMBModel(CMBModel *cmb, PetscReal s, PetscReal *M) {
  *M = cmb->zgrad * (s - cmb->ela);
// FIXME:  add this to formula
/*
  ierr = VecCopy(user->m,Hinitial); CHKERRQ(ierr);
  ierr = VecTrueChop(Hinitial,0.0); CHKERRQ(ierr);
  ierr = VecScale(Hinitial,user->initmagic * user->secpera); CHKERRQ(ierr);
*/
  PetscFunctionReturn(0);
}


PetscErrorCode dMdH_CMBModel(CMBModel *cmb, PetscReal s, PetscReal *dMds) {
  *dMds = cmb->zgrad;
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

