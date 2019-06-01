static const char help[] =
"Solves steady-state, nonlinear ice sheet problem in 2D.  Option prefix ice_.\n"
"The equation is\n"
"       - div (D grad H) - div(W H^{n+2}) = m\n"
"where diffusivity D and pseudo-velocity W (Bueler, 2016) are from the\n"
"nonsliding shallow ice approximation (SIA) flux:\n"
"       D = Gamma H^{n+2} |grad H + grad b|^{n-1}\n"
"       W = - Gamma |grad H + grad b|^{n-1} grad b\n"
"The climatic mass balance m = m(x,y,H) is from one of two models.\n"
"The solution is subject to the constraint\n"
"       H(x,y) >= 0.\n"
"In these equations  H(x,y)  is ice thickness,  b(x,y)  is bed elevation, and\n"
"s = H + b  is surface elevation.  Constants are  n >= 1  and\n"
"Gamma = 2 A (rho g)^n / (n+2)  where A is the ice softness.\n"
"The domain is square  [0,L] x [0,L]  with periodic boundary conditions.\n"
"The equation is discretized by a Q1 structured-grid FVE method (Bueler, 2016).\n"
"Requires SNESVI (-snes_type vinewton{rsls|ssls}) because of constraint;\n"
"defaults to SSLS.\n\n";

/*
1. shows basic success with SSLS; converges to level 7 at least:
   mpiexec -n 4 ./ice -ice_verif -snes_converged_reason -snes_grid_sequence LEV
much more convergence problem without -ice_verif, so:

2. consider making CMB model smooth

3. add CMB to dump and create plotting script (.py)

4. using exact init shows convergence depends strongly on eps for fine grids:
    for LEV in 1 2 3 4 5 6; do ./ice -ice_verif -ice_exact_init -snes_converged_reason -ksp_type gmres -pc_type gamg -da_refine $LEV -ice_eps EPS -snes_max_it 200; done
result: works at all levels if EPS=0.005; then last KSP quite constant but SNES iters significantly variable; which level fails is highly-variable with smaller EPS

5. convergent and nearly optimal in flops *but cheating with exact init*, and *avoiding -snes_grid_sequence*; also works with GMG:
    for LEV in 1 2 3 4 5 6 7 8; do ./ice -da_grid_x 6 -da_grid_y 6 -ice_verif -ice_exact_init -snes_converged_reason -ksp_type gmres -pc_type gamg -da_refine $LEV -snes_type vinewtonrsls -ice_eps 0.005; done

6. visualizing -snes_grid_sequence shows something is VERY WRONG:
    ./ice -da_grid_x 6 -da_grid_y 6 -ice_verif -snes_converged_reason -snes_grid_sequence 3 -snes_type vinewtonrsls -ice_eps 0.1 -snes_monitor_solution draw -draw_pause 1
-snes_grid_sequence bug with periodic BCs? see PETSc issue #300; note RSLS and SSLS act the same; work-around is obvious, and probably wise anyway: use zero Dirichlet boundary conditions
*/

/* see comments on runtime stuff in icet/icet.c, the time-dependent version */

#include <petsc.h>
#include "icecmb.h"

// context is entirely grid-independent info
typedef struct {
    double    secpera,    // number of seconds in a year
              L,          // spatial domain is [0,L] x [0,L]
              g,          // acceleration of gravity
              rho_ice,    // ice density
              n_ice,      // Glen exponent for SIA flux term
              A_ice,      // ice softness
              Gamma,      // coefficient for SIA flux term
              D0,         // representative value of diffusivity (used in regularizing D)
              eps,        // regularization parameter for D
              delta,      // dimensionless regularization for slope in SIA formulas
              lambda;     // amount of upwinding; lambda=0 is none and lambda=1 is "full"
    PetscBool verif,      // use dome formulas if true
              exact_init, // if verif, initialize using dome exact solution
              dump;       // dump state (H,b) after solve
    CMBModel  *cmb;       // defined in cmbmodel.h
} AppCtx;


// compute radius from center of [0,L] x [0,L]
double radialcoord(double x, double y, AppCtx *user) {
  const double xc = x - user->L/2.0,
               yc = y - user->L/2.0;
  return PetscSqrtReal(xc * xc + yc * yc);
}

double DomeCMB(double x, double y, AppCtx *user) {
  const double  domeR  = 750.0e3,  // radius of exact ice sheet (m)
                domeH0 = 3600.0,   // center thickness of exact ice sheet (m)
                n  = user->n_ice,
                pp = 1.0 / n,
                CC = user->Gamma * PetscPowReal(domeH0,2.0*n+2.0)
                         / PetscPowReal(2.0 * domeR * (1.0-1.0/n),n);  // FIXME for n=1?
  double        r, s, tmp1, tmp2;
  r = radialcoord(x, y, user);
  // avoid singularities at center and margin
  if (r < 0.01)
      r = 0.01;
  if (r > domeR - 0.01)
      r = domeR - 0.01;
  s = r / domeR;
  tmp1 = PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp) - 1.0;
  tmp2 = 2.0 * PetscPowReal(s,pp) + PetscPowReal(1.0-s,pp-1.0) * (1.0 - 2.0*s) - 1.0;  // FIXME for n=1 ==> pp = 1?
  return (CC / r) * PetscPowReal(tmp1,n-1.0) * tmp2;  // FIXME for n=1?
}


PetscErrorCode DomeThicknessLocal(DMDALocalInfo *info, double **aH, AppCtx *user) {
  const double   domeR  = 750.0e3,  // radius of exact ice sheet (m)
                 domeH0 = 3600.0,   // center thickness of exact ice sheet (m)
                 n  = user->n_ice,
                 mm = 1.0 + 1.0 / n,
                 qq = n / (2.0 * n + 2.0),
                 CC = domeH0 / PetscPowReal(1.0 - 1.0 / n,qq),  // FIXME for n=1?
                 dx = user->L / (double)(info->mx),
                 dy = user->L / (double)(info->my);
  double         x, y, r, s, tmp;
  int            j, k;

  PetscFunctionBeginUser;
  for (k=info->ys; k<info->ys+info->ym; k++) {
      y = k * dy;
      for (j=info->xs; j<info->xs+info->xm; j++) {
          x = j * dx;
          r = radialcoord(x, y, user);
          // avoid singularities at margin and center
          if (r > domeR - 0.01)
              aH[k][j] = 0.0;
          else {
              if (r < 0.01)
                  r = 0.01;
              s = r / domeR;
              tmp = mm * s - (1.0/n) + PetscPowReal(1.0-s,mm) - PetscPowReal(s,mm);
              aH[k][j] = CC * PetscPowReal(tmp,qq);
          }
      }
  }
  PetscFunctionReturn(0);
}

extern PetscErrorCode SetFromOptionsAppCtx(AppCtx*);
extern PetscErrorCode FormBedLocal(DMDALocalInfo*, int, double**, AppCtx*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double**,
                                        double **, AppCtx*);
//FIXME extern PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, double **,
//                                        Mat, Mat, AppCtx*);

int main(int argc,char **argv) {
  PetscErrorCode      ierr;
  DM                  da;
  SNES                snes;
  KSP                 ksp;
  Vec                 H;
  AppCtx              user;
  CMBModel            cmb;
  DMDALocalInfo       info;
  double              **aH;
  SNESConvergedReason reason;
  int                 snesit,kspit;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = SetFromOptionsAppCtx(&user); CHKERRQ(ierr);
  ierr = SetFromOptions_CMBModel(&cmb,user.secpera);
  user.cmb = &cmb;

  // DMDA for the cell-centered grid
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,
                      DMDA_STENCIL_BOX,
                      3,3, PETSC_DECIDE,PETSC_DECIDE,
                      1, 1,        // dof=1, stencilwidth=1
                      NULL,NULL,&da);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(da, &user);CHKERRQ(ierr);

  // create and configure the SNES to solve a NCP/VI at each step
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
//FIXME
//  ierr = DMDASNESSetJacobianLocal(da,
//               (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONSSLS); CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // set up initial iterate
  ierr = DMCreateGlobalVector(da,&H);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)H,"H"); CHKERRQ(ierr);
  if (user.exact_init) {
      ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da,H,&aH); CHKERRQ(ierr);
      ierr = DomeThicknessLocal(&info,aH,&user); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(da,H,&aH); CHKERRQ(ierr);
  } else {
      ierr = VecSet(H,0.0); CHKERRQ(ierr);
  }

  // solve
  ierr = SNESSolve(snes,NULL,H); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  if (reason <= 0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "WARNING: SNES not converged ... use -snes_converged_reason to check\n"); CHKERRQ(ierr);
  }

  // get solution & DM on fine grid (which may have changed) after solve
  ierr = VecDestroy(&H); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&da); CHKERRQ(ierr); /* do not destroy da */
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&H); CHKERRQ(ierr); /* do not destroy H */
  ierr = PetscObjectSetName((PetscObject)H,"H"); CHKERRQ(ierr);

  // compute performance measures; note utility of reporting last grid,
  //   last snesit/kspit when doing -snes_grid_sequence
  ierr = SNESGetIterationNumber(snes,&snesit); CHKERRQ(ierr);  // 
  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&kspit); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "done on %d x %d grid ... SNES iters = %d, last KSP iters = %d\n",
      info.mx,info.my,snesit,kspit); CHKERRQ(ierr);

  // dump state (H,b) if requested
  if (user.dump) {
      char           filename[1024];
      PetscViewer    viewer;
      Vec            b;
      double         **ab;
      ierr = VecDuplicate(H,&b); CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)b,"b"); CHKERRQ(ierr);
      if (user.verif) {
          ierr = VecSet(b,0.0); CHKERRQ(ierr);
      } else {
          ierr = DMDAVecGetArray(da,b,&ab); CHKERRQ(ierr);
          ierr = FormBedLocal(&info,0,ab,&user); CHKERRQ(ierr);
          ierr = DMDAVecRestoreArray(da,b,&ab); CHKERRQ(ierr);
      }
      ierr = sprintf(filename,"ice_%dx%d.dat",info.mx,info.my);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"writing PETSC binary file %s ...\n",filename); CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
      ierr = VecView(b,viewer); CHKERRQ(ierr);
      ierr = VecView(H,viewer); CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
      VecDestroy(&b);
  }

  // compute error in verification case
  if (user.verif) {
      Vec Hexact;
      double infnorm, onenorm;
      ierr = VecDuplicate(H,&Hexact); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da,Hexact,&aH); CHKERRQ(ierr);
      ierr = DomeThicknessLocal(&info,aH,&user); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(da,Hexact,&aH); CHKERRQ(ierr);
      ierr = VecAXPY(H,-1.0,Hexact); CHKERRQ(ierr);    // H <- H + (-1.0) Hexact
      VecDestroy(&Hexact);
      ierr = VecNorm(H,NORM_INFINITY,&infnorm); CHKERRQ(ierr);
      ierr = VecNorm(H,NORM_1,&onenorm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "numerical errors: |H-Hexact|_inf = %.3f, |H-Hexact|_average = %.3f\n",
          infnorm,onenorm/(double)(info.mx*info.my)); CHKERRQ(ierr);
  }

  SNESDestroy(&snes);
  return PetscFinalize();
}

// FIXME  put this back in main() so only actual context params need to be in AppCtx
// (e.g. exact_init, dump do not need to be in AppCtx)
PetscErrorCode SetFromOptionsAppCtx(AppCtx *user) {
  PetscErrorCode ierr;

  user->secpera    = 31556926.0;  // number of seconds in a year
  user->L          = 1800.0e3;    // m; compare domeR=750.0e3 radius
  user->g          = 9.81;        // m/s^2
  user->rho_ice    = 910.0;       // kg/m^3
  user->n_ice      = 3.0;         // Glen exponent
  user->A_ice      = 3.1689e-24;  // 1/(Pa^3 s); EISMINT I value
  user->D0         = 1.0;         // m^2 / s
  user->eps        = 0.001;
  user->delta      = 1.0e-4;
  user->lambda     = 0.25;
  user->verif      = PETSC_FALSE;
  user->exact_init = PETSC_FALSE;
  user->dump       = PETSC_FALSE;
  user->cmb        = NULL;
  // user->Gamma is derived below

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
  ierr = PetscOptionsBool(
      "-dump", "save final state (H, b)",
      "ice.c",user->dump,&user->dump,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-eps", "dimensionless regularization for diffusivity D",
      "ice.c",user->eps,&user->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-exact_init", "initialize with dome exact solution",
      "ice.c",user->exact_init,&user->exact_init,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-L", "side length of domain in meters",
      "ice.c",user->L,&user->L,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-lambda", "amount of upwinding; lambda=0 is none and lambda=1 is full",
      "ice.c",user->lambda,&user->lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-n", "value of Glen exponent n",
      "ice.c",user->n_ice,&user->n_ice,NULL);CHKERRQ(ierr);
  if (user->n_ice < 1.0) {
      SETERRQ1(PETSC_COMM_WORLD,1,
          "ERROR: n = %f not allowed ... n >= 1 is required\n",user->n_ice);
  }
  ierr = PetscOptionsReal(
      "-rho", "ice density in units kg m3",
      "ice.c",user->rho_ice,&user->rho_ice,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-verif", "use dome exact solution for verification",
      "ice.c",user->verif,&user->verif,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // derived constant computed after other ice properties are set
  user->Gamma = 2.0 * PetscPowReal(user->rho_ice*user->g,user->n_ice) 
                    * user->A_ice / (user->n_ice+2.0);

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

#if 0
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
#endif

/* Pseudo-velocity from bed slope:  W = - sigma * grad b. */
static Grad W(double sigma, Grad gb) {
    Grad W;
    W.x = - sigma * gb.x;
    W.y = - sigma * gb.y;
    return W;
}

#if 0
/* Derivative of pseudo-velocity W with respect to nodal value H_l. */
static Grad DWDl(double dsigmadl, Grad gb) {
    Grad dWdl;
    dWdl.x = - dsigmadl * gb.x;
    dWdl.y = - dsigmadl * gb.y;
    return dWdl;
}
#endif

/* DCS = diffusivity from the continuation scheme:
     D(eps) = (1-eps) sigma H^{n+2} + eps D_0
so D(1)=D_0 and D(0)=sigma H^{n+2}. */
static double DCS(double sigma, double H, const AppCtx *user) {
  return (1.0 - user->eps) * sigma * PetscPowReal(PetscAbsReal(H),user->n_ice+2.0)
         + user->eps * user->D0;
}


#if 0
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
#endif

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


#if 0
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
#endif

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


#if 0
static double dfieldatpt(int l, double xi, double eta) {
  const double x[4] = { 1.0-xi,      xi,  xi, 1.0-xi},
               y[4] = {1.0-eta, 1.0-eta, eta,    eta};
  return x[l] * y[l];
}
#endif


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


#if 0
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
#endif

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


/* FormFunctionLocal  =  call-back by SNES using DMDA info.

Evaluates residual FF on local process patch:
   FF_{j,k} = \int_{\partial V_{j,k}} \mathbf{q} \cdot \mathbf{n}
              - m_{j,k} \Delta x \Delta y
where V_{j,k} is the control volume centered at (x_j,y_k).

Regarding indexing locations along the boundary of the control volume where
flux is evaluated, this figure shows the control volume centered at (x_j,y_k)
and the four elements it meets.  Quadrature uses 8 points on the boundary of
the control volume, numbered s=0,...,7:

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
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **aH,
                                        double **FF, AppCtx *user) {
  PetscErrorCode  ierr;
  const double    dx = user->L / (double)(info->mx),
                  dy = user->L / (double)(info->my);
  // coefficients of quadrature evaluations along the boundary of the control volume in M*
  const double    coeff[8] = {dy/2, dx/2, dx/2, -dy/2, -dy/2, -dx/2, -dx/2, dy/2};
  const PetscBool upwind = (user->lambda > 0.0);
  const double    upmin = (1.0 - user->lambda) * 0.5,
                  upmax = (1.0 + user->lambda) * 0.5;
  int             c, j, k, s;
  double          H, Hup, lxup, lyup, **aqquad[4], **ab, DSIA_ckj, qSIA_ckj,
                  M, x, y;
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

  // get bed elevation b(x,y) on this grid
  ierr = DMGetLocalVector(info->da, &b); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da,b,&ab); CHKERRQ(ierr);
  if (user->verif) {
      ierr = VecSet(b,0.0); CHKERRQ(ierr);
  } else {
      ierr = FormBedLocal(info,1,ab,user); CHKERRQ(ierr);  // get stencil width
  }

  // working space for fluxes; see text for face location of flux evaluation
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
          }
      }
  }

  // loop over nodes, not including ghosts, to get function F(H) from quadature over
  // s = 0,1,...,7 points on boundary of control volume (rectangle) around node
  for (k=info->ys; k<info->ys+info->ym; k++) {
      for (j=info->xs; j<info->xs+info->xm; j++) {
          // climatic mass balance
          if (user->verif) {
              x = j * dx;
              y = k * dy;
              M = DomeCMB(x,y,user);
          } else {
              M = M_CMBModel(user->cmb,ab[k][j] + aH[k][j]);  // s=b+H is surface elevation
          }
          FF[k][j] = - M * dx * dy;
          // now add integral over control volume boundary using two
          // quadrature points on each side
          for (s=0; s<8; s++)
              FF[k][j] += coeff[s] * aqquad[ce[s]][k+ke[s]][j+je[s]];
      }
  }


  // restore working space and bed
  for (c = 0; c < 4; c++) {
      ierr = DMDAVecRestoreArray(info->da,qquad[c],&(aqquad[c])); CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(info->da, &(qquad[c])); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(info->da,b,&ab); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(info->da, &b); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#if 0

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

#endif

