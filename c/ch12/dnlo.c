static const char help[] = "\n"
"Solves doubly-nonlinear obstacle problems in 2D.  Option prefix dn_.\n"
"The PDE (interior condition) of such problems is\n"
"    - div (C u^q |grad(u+b)|^{p-2} grad(u+b)) = f(u,x,y)\n"
"where the solution u(x,y) is subject to a obstacle constraint\n"
"    u >= psi\n"
"Here psi(x,y) and b(x,y) are given functions and C>0, q >= 0, and p > 1 are\n"
"constants.  Method uses diffusion-advection (Bueler, 2016) decomposition\n"
"    - div (D grad u) + div(W u^q) = f\n"
"where\n"
"    D = C u^q |grad(u+b)|^{p-2}       is the diffusivity and\n"
"    W = - C |grad(u+b)|^{p-2} grad b  is the pseudo-velocity.\n"
"The equation is discretized by a Q1 structured-grid FVE method (Bueler, 2016).\n"
"Requires SNESVI (-snes_type vinewton{rsls|ssls}) because of the constraint.\n"
"The domain is square with zero Dirichlet boundary conditions.\n\n"
"Default problem is classical obstacle (-dnl_problem obstacle) with domain\n"
"(-2,2)^2, C = 1, q = 0, p = 2, b = 0, f = 0, and psi(x,y) giving a \n"
"hemispherical obstacle.\n\n"
"Can solve a steady-state ice sheet problem (-dnl_problem ice) in 2D\n"
"in which  u = H  is ice thickness,  b  is bed elevation, and  s = H + b  is\n"
"ice surface elevation.  In that case  p = n+1  where  n >= 1  is the Glen\n"
"exponent, q = n+2, and  C  is computed using the ice softness, ice density,\n"
"and gravity.  In the ice case  Q = - D grad u + W u^q  is the nonsliding\n"
"shallow ice approximation (SIA) flux and the climatic mass balance\n"
"f = m(H,x,y) is from one of two models.  See ice.h.\n\n";

/*
1. looks reasonable with RSLS:
./dnlo -snes_monitor -ksp_converged_reason -pc_type mg -snes_grid_sequence 3

2. looks like it works with SSLS:
./dnlo -snes_type vinewtonssls -snes_monitor -snes_monitor_solution draw -draw_pause 1 -da_refine 1
./obstacle -da_refine 1 -snes_monitor_solution draw -draw_pause 1 -snes_monitor -snes_type vinewtonssls
*/

/* FIXME: OLD RUNS:
1. shows basic success with SSLS but DIVERGES AT LEVEL 4:
   mpiexec -n 4 ./ice -ice_verif -snes_converged_reason -snes_grid_sequence LEV

2. consider making CMB model smooth

3. add CMB to dump and create plotting script (.py)

4. using exact init shows convergence depends strongly on eps for fine grids:
    for LEV in 1 2 3 4 5; do ./ice -ice_verif -ice_exact_init -snes_converged_reason -ksp_type gmres -pc_type gamg -da_refine $LEV -ice_eps EPS; done
result:
  (a) works at all levels if EPS=0.005; last KSP somewhat constant but SNES iters growing
  (b) fails on level 3 if EPS=0.003,0.002

5. convergent and nearly optimal GMG in flops *but cheating with exact init*, and *avoiding -snes_grid_sequence* and *significant eps=0.01 regularization*:
    for LEV in 1 2 3 4 5 6 7 8; do ./ice -ice_verif -ice_exact_init -snes_converged_reason -ksp_type gmres -pc_type mg -da_refine $LEV -snes_type vinewtonrsls -ice_eps 0.01; done

6. visualizing -snes_grid_sequence:
    ./ice -ice_verif -snes_grid_sequence 2 -ice_eps 0.005 -snes_converged_reason -snes_monitor_solution draw
(was -snes_grid_sequence bug with periodic BCs? see PETSc issue #300)

8. even seems to work in parallel:
    mpiexec -n 4 ./ice -ice_verif -snes_grid_sequence 5 -ice_eps 0.005 -snes_converged_reason -snes_monitor_solution draw

9. same outcome with -ice_exact_init and -da_refine 5
    mpiexec -n 4 ./ice -ice_verif -da_refine 5 -ice_eps 0.005 -snes_converged_reason -snes_monitor_solution draw -ice_exact_init

10. unpredictable response to changing -snes_linesearch_type bt|l2|basic  (cp seems rarely to work)
*/

/* see comments on runtime stuff in icet/icet.c, the time-dependent version */

#include <petsc.h>
#include "ice.h"

typedef struct {
    double    C,          // coefficient
              q,          // power on u (porous medium type degeneracy)
              p,          // p-Laplacian power (|grad u|^{p-2} degeneracy)
              D0,         // representative value of diffusivity (used for regularizing D)
              eps,        // regularization parameter for diffusivity D
              delta,      // dimensionless regularization for |grad(u+b)| term
              lambda;     // amount of upwinding; lambda=0 is none and lambda=1 is "full"
    PetscBool check_admissible; // check admissibility at start of FormFunctionLocal()
    double    (*psi)(double,double), // evaluate obstacle  psi(x,y)  at point
              (*bed)(double,double); // evaluate bed elevation  b(x,y)  at point
} AppCtx;

// z = hemisphere(x,y) is same obstacle as in obstacle.c
double hemisphere(double x, double y) {
    const double  r = x * x + y * y,  // FIXME this is from (0,0)
                  r0 = 0.9,
                  psi0 = PetscSqrtReal(1.0 - r0*r0),
                  dpsi0 = - r0 / psi0;
    if (r <= r0) {
        return PetscSqrtReal(1.0 - r);
    } else {
        return psi0 + dpsi0 * (r - r0);
    }
}

// obstacle is zero in ice sheet case
double zero(double x, double y) {
    return 0.0;
}

typedef enum {OBSTACLE, ICE} ProblemType;
static const char *ProblemTypes[] = {"obstacle", "ice",
                                     "ProblemType", "", NULL};

typedef enum {ZERO, ROLLING} IceBedType;
static const char *IceBedTypes[] = {"zero", "rolling",
                                     "IceBedType", "", NULL};

extern PetscErrorCode FormBounds(SNES,Vec,Vec);
extern PetscErrorCode FormField(DM, double (*)(double,double), Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double**, double**, AppCtx*);

int main(int argc,char **argv) {
  PetscErrorCode      ierr;
  DM                  da;
  SNES                snes;
  KSP                 ksp;
  Vec                 u;
  AppCtx              user;
  ProblemType         problem = OBSTACLE;
  IceBedType          icebed = ZERO;
  PetscBool           exact_init = PETSC_FALSE,
                      dump = PETSC_FALSE;
  DMDALocalInfo       info;
  SNESConvergedReason reason;
  int                 snesit,kspit;

  PetscInitialize(&argc,&argv,(char*)0,help);

  user.C          = 1.0;
  user.q          = 0.0;
  user.p          = 2.0;
  user.D0         = 0.0;         // m^2 / s
  user.eps        = 0.0;
  user.delta      = 1.0e-4;
  user.lambda     = 0.25;
  user.check_admissible = PETSC_FALSE;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"dn_","options to dnlo","");CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-check_admissible", "check admissibility of iterate at start of residual evaluation FormFunctionLocal()",
      "dnlo.c",user.check_admissible,&user.check_admissible,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-D0", "representative value of diffusivity (used in regularizing D) in units m2 s-1",
      "dnlo.c",user.D0,&user.D0,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-delta", "dimensionless regularization for slope",
      "dnlo.c",user.delta,&user.delta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-dump", "save final state (u, psi, b) in file dnlo_MXxMY.dat",
      "dnlo.c",dump,&dump,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-eps", "dimensionless regularization for diffusivity D",
      "dnlo.c",user.eps,&user.eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool(
      "-exact_init", "initialize with exact solution (only possible if FIXME)",
      "dnlo.c",exact_init,&exact_init,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-ice_bed","type of bed elevation map to use with -dnl_problem ice",
      "dnlo.c",IceBedTypes,
      (PetscEnum)(icebed),(PetscEnum*)&(icebed),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-lambda", "amount of upwinding; lambda=0 is none and lambda=1 is full",
      "dnlo.c",user.lambda,&user.lambda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-p", "p-Laplacian exponent",
      "dnlo.c",user.p,&user.p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-problem","problem type",
      "dnlo.c",ProblemTypes,
      (PetscEnum)(problem),(PetscEnum*)&(problem),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsReal(
      "-q", "porous medium type exponent",
      "dnlo.c",user.q,&user.q,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  // DMDA for the cell-centered grid
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
                      DMDA_STENCIL_BOX,
                      5,5, PETSC_DECIDE,PETSC_DECIDE,
                      1, 1,        // dof=1, stencilwidth=1
                      NULL,NULL,&da);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(da, &user);CHKERRQ(ierr);

  // set domain, obstacle, and b (bed)
  user.bed = &zero;
  if (problem == ICE) {
      const double L = 1800.0e3;
      ierr = DMDASetUniformCoordinates(da,0.0,L,0.0,L,-1.0,-1.0);CHKERRQ(ierr);
      user.psi = &zero;
      if (icebed == ROLLING) {
          user.bed = &rollingbed;
      }
  } else { // problem == OBSTACLE
      ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,-1.0,-1.0);CHKERRQ(ierr);
      user.psi = &hemisphere;
  }

  // create and configure the SNES to solve a NCP/VI at each step
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS); CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // set up initial iterate
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u"); CHKERRQ(ierr);
  ierr = VecSet(u,0.0); CHKERRQ(ierr);
/* FIXME
  if (exact_init) {
      double **aH;
      ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(da,H,&aH); CHKERRQ(ierr);
      ierr = DomeThicknessLocal(&info,aH,&user); CHKERRQ(ierr);
      ierr = DMDAVecRestoreArray(da,H,&aH); CHKERRQ(ierr);
  }
*/

  // solve
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "solving problem %s using q = %.3f and p = %.3f ...\n",
      ProblemTypes[problem],user.q,user.p); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  if (reason <= 0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "WARNING: SNES not converged ... use -snes_converged_reason to check\n");
          CHKERRQ(ierr);
  }

  // get solution & DM on fine grid (which may have changed) after solve
  ierr = VecDestroy(&u); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&da); CHKERRQ(ierr); /* do not destroy da */
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr); /* do not destroy H */
  ierr = PetscObjectSetName((PetscObject)u,"u"); CHKERRQ(ierr);

  // compute performance measures; note it is useful to report last grid and
  //   last snesit/kspit when doing -snes_grid_sequence
  ierr = SNESGetIterationNumber(snes,&snesit); CHKERRQ(ierr);  // 
  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&kspit); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "done on %d x %d grid (SNES iters = %d, last KSP iters = %d)\n",
      info.mx,info.my,snesit,kspit); CHKERRQ(ierr);

  // dump state (u,psi,b) if requested
  if (dump) {
      char           filename[1024];
      PetscViewer    viewer;
      Vec            psi, b;
      ierr = DMGetGlobalVector(da,&psi);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(da,&b);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)psi,"psi"); CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)b,"b"); CHKERRQ(ierr);
      ierr = FormField(da,user.psi,psi); CHKERRQ(ierr);
      ierr = FormField(da,user.bed,b); CHKERRQ(ierr);
      ierr = sprintf(filename,"dnlo_%dx%d.dat",info.mx,info.my);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"writing PETSC binary file %s ...\n",
          filename); CHKERRQ(ierr);
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,
          &viewer); CHKERRQ(ierr);
      ierr = VecView(u,viewer); CHKERRQ(ierr);
      ierr = VecView(psi,viewer); CHKERRQ(ierr);
      ierr = VecView(b,viewer); CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(da,&psi);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(da,&b);CHKERRQ(ierr);
  }

/* FIXME
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
*/

  SNESDestroy(&snes);
  return PetscFinalize();
}

// bounds: psi <= u < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
    PetscErrorCode ierr;
    DM            da;
    DMDALocalInfo info;
    AppCtx        *user;
    int           i, j;
    double        **aXl, dx, dy, x, y, xymin[2], xymax[2];
    ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = DMDAGetBoundingBox(info.da,xymin,xymax); CHKERRQ(ierr);
    dx = (xymax[0] - xymin[0]) / (info.mx - 1);
    dy = (xymax[1] - xymin[1]) / (info.my - 1);
    ierr = SNESGetApplicationContext(snes,&user); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Xl, &aXl);CHKERRQ(ierr);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = xymin[1] + j * dy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = xymin[0] + i * dx;
            aXl[j][i] = (*(user->psi))(x,y);
        }
    }
    ierr = DMDAVecRestoreArray(da, Xl, &aXl);CHKERRQ(ierr);
    ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormField(DM da, double (*f)(double,double), Vec v) {
    PetscErrorCode ierr;
    DMDALocalInfo info;
    int           i, j;
    double        **av, dx, dy, x, y, xymin[2], xymax[2];
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = DMDAGetBoundingBox(da,xymin,xymax); CHKERRQ(ierr);
    dx = (xymax[0] - xymin[0]) / (info.mx - 1);
    dy = (xymax[1] - xymin[1]) / (info.my - 1);
    ierr = DMDAVecGetArray(da, v, &av);CHKERRQ(ierr);
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = xymin[1] + j * dy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = xymin[0] + i * dx;
            av[j][i] = (*f)(x,y);
        }
    }
    ierr = DMDAVecRestoreArray(da, v, &av);CHKERRQ(ierr);
    return 0;
}

// value of gradient at a point
typedef struct {
    double x,y;
} Grad;

/* We first write the flux as
    Q = - u^q sigma(|grad u + grad b|) (grad u + grad b)
where sigma is the function for the slope-dependent part
    sigma(z) = C z^{p-2}.
and write
    D = u^q sigma(|grad u + grad b|)
so that Q = - D (grad u + grad b).  Then we split diffusive
and advective parts as
    Q = - D grad u + W u^q
where
    W = - sigma grad b.            */

/* The number sigma(|grad u + grad b|). */
static double sigma(Grad gu, Grad gb, const AppCtx *user) {
    const double sx = gu.x + gb.x,
                 sy = gu.y + gb.y,
                 slopesqr = sx * sx + sy * sy + user->delta * user->delta;
    return user->C * PetscPowReal(slopesqr,(user->p-2.0)/2);
}

/* Pseudo-velocity from bed slope:  W = - sigma * grad b. */
static Grad W(double sigma, Grad gb) {
    Grad W;
    W.x = - sigma * gb.x;
    W.y = - sigma * gb.y;
    return W;
}

/* DCS = diffusivity from the continuation scheme:
     D(eps) = (1-eps) u^q sigma + eps D_0
so D(1)=D_0 and D(0)=u^q sigma. */
static double DCS(double sigma, double u, const AppCtx *user) {
  return (1.0 - user->eps) * sigma * PetscPowReal(PetscAbsReal(u),user->q)
         + user->eps * user->D0;
}

/* Flux component Q_x or Q_y on a general bed. */
PetscErrorCode DNLflux(Grad gu, Grad gb, double u, double uup, PetscBool xdir,
                       double *D, double *Q, const AppCtx *user) {
  const double mysig = sigma(gu,gb,user),
               myD   = DCS(mysig,u,user);
  const Grad   myW   = W(mysig,gb);
  PetscFunctionBeginUser;
  if (D) {
      *D = myD;
  }
  if (xdir && Q) {
      *Q = - myD * gu.x + myW.x * PetscPowReal(PetscAbsReal(uup),user->q);
  } else {
      *Q = - myD * gu.y + myW.y * PetscPowReal(PetscAbsReal(uup),user->q);
  }
  PetscFunctionReturn(0);
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

// coefficients of quadrature evaluations along the boundary of the control volume in M*
static PetscErrorCode fluxcoeffs(double dx, double dy, double *c) {
  c[0] = dy/2;
  c[1] = dx/2;
  c[2] = dx/2;
  c[3] = -dy/2;
  c[4] = -dy/2;
  c[5] = -dx/2;
  c[6] = -dx/2;
  c[7] = dy/2;
  return 0;
}

/* FormFunctionLocal  =  call-back by SNES using DMDA info.

Evaluates residual FF on local process patch:
   FF_{j,k} = \int_{\partial V_{j,k}} \mathbf{Q} \cdot \mathbf{n}
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
the value  (aQquad[c])[k][j] for c=0,1,2,3 is an x-component at "*" and
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
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **auin,
                                 double **FF, AppCtx *user) {
  PetscErrorCode  ierr;
  const PetscBool upwind = (user->lambda > 0.0);
  const double    upmin = (1.0 - user->lambda) * 0.5,
                  upmax = (1.0 + user->lambda) * 0.5;
  int             c, j, k, s;
  double          xymin[2], xymax[2], dx, dy, x, y, coeff[8], bb[4],
                  u, uup, lxup, lyup, **aQquad[4], **au, Q_ckj, M;
  Grad            gu, gb;
  Vec             Qquad[4], ucopy;

  PetscFunctionBeginUser;
  ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
  dx = (xymax[0] - xymin[0]) / (info->mx - 1);
  dy = (xymax[1] - xymin[1]) / (info->my - 1);
  ierr = fluxcoeffs(dx,dy,coeff); CHKERRQ(ierr);

  // copy and set boundary conditions to zero
  ierr = DMGetLocalVector(info->da, &ucopy); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da,ucopy,&au); CHKERRQ(ierr);
  for (k = info->ys-1; k <= info->ys + info->ym; k++) {
      for (j = info->xs-1; j <= info->xs + info->xm; j++) {
          if (j < 0 || j > info->mx-1 || k < 0 || k > info->my-1)
              continue;
          if (user->check_admissible && auin[k][j] < 0.0) {
              SETERRQ3(PETSC_COMM_WORLD,1,
                       "ERROR: non-admissible value u[k][j] = %.3e < 0.0 at j,k = %d,%d\n",
                       auin[k][j],j,k);
          }
          if (j == 0 || j == info->mx-1 || k == 0 || k == info->my-1) {
              if (   j >= info->xs && j < info->xs+info->xm
                  && k >= info->ys && k < info->ys+info->ym)
                  FF[k][j] = auin[k][j];   // FIXME scaling?
              au[k][j] = 0.0;
          } else
              au[k][j] = auin[k][j];
      }
  }

  // working space for fluxes; see text for face location of flux evaluation
  for (c = 0; c < 4; c++) {
      ierr = DMGetLocalVector(info->da, &(Qquad[c])); CHKERRQ(ierr);
      ierr = DMDAVecGetArray(info->da,Qquad[c],&(aQquad[c])); CHKERRQ(ierr);
  }

  // loop over locally-owned elements, including ghosts, to get fluxes q at
  // c = 0,1,2,3 points in element;  note start at (xs-1,ys-1)
  for (k = info->ys-1; k < info->ys + info->ym; k++) {
      y = xymin[1] + k * dy;
      for (j = info->xs-1; j < info->xs + info->xm; j++) {
          if (j < 0 || j >= info->mx-1 || k < 0 || k >= info->my-1)
              continue;
          x = xymin[0] + j * dx;
          bb[0] = (*(user->bed))(x,y);
          bb[1] = (*(user->bed))(x+dx,y);
          bb[2] = (*(user->bed))(x+dx,y+dy);
          bb[3] = (*(user->bed))(x,y+dy);
          for (c=0; c<4; c++) {
              // u and grad u at quadrature point
              u  = fieldatptArray(j,k,locx[c],locy[c],au);
              gu = gradfatptArray(j,k,locx[c],locy[c],dx,dy,au);
              // grad b at quadrature point
              gb = gradfatpt(locx[c],locy[c],dx,dy,bb);
              if (upwind) {
                  if (xdire[c] == PETSC_TRUE) {
                      lxup = (gb.x <= 0.0) ? upmin : upmax;
                      lyup = locy[c];
                  } else {
                      lxup = locx[c];
                      lyup = (gb.y <= 0.0) ? upmin : upmax;
                  }
                  uup = fieldatptArray(j,k,lxup,lyup,au);
              } else
                  uup = u;
              ierr = DNLflux(gu,gb,u,uup,xdire[c],NULL,&Q_ckj,user); CHKERRQ(ierr);
              aQquad[c][k][j] = Q_ckj;
          }
      }
  }

  // loop over nodes, not including ghosts, to get function F(H) from quadature over
  // s = 0,1,...,7 points on boundary of control volume (rectangle) around node
  for (k=info->ys; k<info->ys+info->ym; k++) {
      for (j=info->xs; j<info->xs+info->xm; j++) {
          if (j == 0 || j == info->mx-1 || k == 0 || k == info->my-1)
              continue;
/* FIXME
          // climatic mass balance
          if (user->verif) {
              x = j * dx;
              y = k * dy;
              M = DomeCMB(x,y,user);
          } else {
              M = M_CMBModel(user->cmb,ab[k][j] + aH[k][j]);  // s=b+H is surface elevation
          }
*/
          M = 0.0;  // FIXME
          FF[k][j] = - M * dx * dy;
          // now add integral over control volume boundary using two
          // quadrature points on each side
          for (s=0; s<8; s++)
              FF[k][j] += coeff[s] * aQquad[ce[s]][k+ke[s]][j+je[s]];
      }
  }


  // restore working space
  for (c = 0; c < 4; c++) {
      ierr = DMDAVecRestoreArray(info->da,Qquad[c],&(aQquad[c])); CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(info->da, &(Qquad[c])); CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArray(info->da,ucopy,&au); CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(info->da, &ucopy); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

