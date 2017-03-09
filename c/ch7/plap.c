static char help[] = "Solves a p-Laplacian equation in 2D using Q^1 FEM:\n"
"   - div (a0 |grad u|^{p-2} grad u) + c u = f\n"
"where a0 is constant and functions c(x,y), f(x,y) are given.  Uses periodic\n"
"boundary conditions on the unit square.  Implements an objective function\n"
"and a residual (= gradient = weak form), but no Jacobian.  Allows optional\n"
"regularization of the nonlinearity.\n"
"Default case with manufactured solution has a=a0, c=c0, and p=4.  Default\n"
"quadrature degree is n=2.  Run as one of:\n"
"   ./plap -snes_fd_color                   [default]\n"
"   ./plap -snes_mf\n"
"   ./plap -snes_fd                         [does not scale]\n"
"   ./plap -snes_fd_function -snes_fd_color [does not scale]\n\n";

#include <petsc.h>

//STARTCTX
typedef struct {
    double  p, eps, a0, c0;
    int     quaddegree;
} PLapCtx;

PetscErrorCode ConfigureCtx(PLapCtx *user) {
    PetscErrorCode ierr;
    user->p = 4.0;
    user->eps = 0.0;
    user->a0 = 1.0;
    user->c0 = 1.0;
    user->quaddegree = 2;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"plap_",
                       "p-laplacian solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                      NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
    if (user->p < 1.0) { SETERRQ(PETSC_COMM_WORLD,1,"p >= 1 required"); }
    ierr = PetscOptionsReal("-eps","regularization parameter eps",
                      NULL,user->eps,&(user->eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-a0","coefficient of nonlinear (p-laplacian) term",
                      NULL,user->a0,&(user->a0),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-c0","coefficient of zeroth-order linear term",
                      NULL,user->c0,&(user->c0),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                     NULL,user->quaddegree,&(user->quaddegree),NULL); CHKERRQ(ierr);
    if ((user->quaddegree < 1) || (user->quaddegree > 3)) {
        SETERRQ(PETSC_COMM_WORLD,2,"quadrature degree n=1,2,3 only"); }
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}
//ENDCTX

//STARTDATA
double Psi(double z) {
    return sin(2.0 * PETSC_PI * z);
}

double dPsi(double z) {
    return 2.0 * PETSC_PI * cos(2.0 * PETSC_PI * z);
}

double ddPsi(double z) {
    return - 4.0 * PETSC_PI * PETSC_PI * sin(2.0 * PETSC_PI * z);
}

double Phi(double z, double p) {
    return PetscPowReal(PetscAbsReal(z),p - 2.0) * z;
}

double dPhi(double z, double p) {
    return (p - 1.0) * PetscPowReal(PetscAbsReal(z),p - 2.0);
}

double Uexact(double x, double y, PLapCtx *user) {
    return Psi(x + y);
}

double Frhs(double x, double y, PLapCtx *user) {
    const double z = x + y,
                 CC = PetscPowReal(2.0, user->p / 2.0);
    return - user->a0 * CC * dPhi(dPsi(z),user->p) * ddPsi(z)
           + user->c0 * Psi(z);
}

// coefficient of linear term in PDE
double Ccoeff(double x, double y, PLapCtx *user) {
    return user->c0;
}
//ENDDATA

PetscErrorCode getUexact(DMDALocalInfo *info, Vec uexact, PLapCtx* user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / info->mx,  hy = 1.0 / info->my;
    int          i, j;
    double       x, y, **auexact;
    ierr = DMDAVecGetArray(info->da, uexact, &auexact);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;
            auexact[j][i] = Uexact(x,y,user);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uexact, &auexact);CHKERRQ(ierr);
    return 0;
}

//STARTGRADTOOLS
typedef struct {
    double  xi, eta;
} gradRef;

double GradInnerProd(DMDALocalInfo *info, gradRef du, gradRef dv) {
    const double hx = 1.0 / info->mx,  hy = 1.0 / info->my,
                 cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

double GradPow(DMDALocalInfo *info, gradRef du, double p, double eps) {
    return PetscPowScalar(GradInnerProd(info,du,du) + eps * eps, p / 2.0);
}
//ENDGRADTOOLS

//STARTFEM
static double xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
              etaL[4] = { 1.0,  1.0, -1.0, -1.0};

double chi(int L, double xi, double eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

gradRef dchi(int L, double xi, double eta) {
    gradRef result;
    result.xi  = 0.25 * xiL[L]  * (1.0 + etaL[L] * eta);
    result.eta = 0.25 * etaL[L] * (1.0 + xiL[L]  * xi);
    return result;
}

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[4], double xi, double eta) {
    double sum = 0.0;
    int    L;
    for (L=0; L<4; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

// evaluate partial derivs of v(xi,eta) on reference element
gradRef deval(const double v[4], double xi, double eta) {
    gradRef sum = {0.0,0.0}, tmp;
    int     L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi  += v[L] * tmp.xi;
        sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

static double
    zq[3][3] = { {0.0,NAN,NAN},
                 {-0.577350269189626,0.577350269189626,NAN},
                 {-0.774596669241483,0.0,0.774596669241483} },
    wq[3][3] = { {2.0,NAN,NAN},
                 {1.0,1.0,NAN},
                 {0.555555555555556,0.888888888888889,0.555555555555556} };
//ENDFEM

//STARTOBJECTIVE
double ObjIntegrandRef(DMDALocalInfo *info, double xi, double eta,
                       const double u[4], const double c[4], const double f[4],
                       PLapCtx *user) {
    const gradRef du = deval(u,xi,eta);
    const double  uu = eval(u,xi,eta);
    return (user->a0 / user->p) * GradPow(info,du,user->p,user->eps)
           + (eval(c,xi,eta) / 2.0) * uu * uu - eval(f,xi,eta) * uu;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / info->mx,  hy = 1.0 / info->my;
  const int    n = user->quaddegree;
  double       x, y, lobj = 0.0;
  int          i,j,r,s;
  MPI_Comm     com;

  // loop over all elements: get unique integral contribution from each element
  for (j = info->ys; j < info->ys+info->ym; j++) {
      y = j * hy;
      for (i = info->xs; i < info->xs+info->xm; i++) {
          x = i * hx;   // (x,y) is location of (i,j) node; lower-left corner on element
          const double u[4] = {au[j+1][i+1], au[j+1][i], au[j][i], au[j][i+1]},
                       c[4] = {Ccoeff(x+hx,y+hy,user), Ccoeff(x,   y+hy,user),
                               Ccoeff(x,   y,   user), Ccoeff(x+hx,y,   user)},
                       f[4] = {Frhs(x+hx,y+hy,user), Frhs(x,   y+hy,user),
                               Frhs(x,   y,   user), Frhs(x+hx,y,   user)};
          // loop over quadrature points
          for (r=0; r<n; r++) {
              for (s=0; s<n; s++) {
                  lobj += wq[n-1][r] * wq[n-1][s]
                          * ObjIntegrandRef(info,zq[n-1][r],zq[n-1][s],
                                            u,c,f,user);
              }
          }
      }
  }
  lobj *= 0.25 * hx * hy;

  ierr = PetscObjectGetComm((PetscObject)(info->da),&com); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
double FunIntegrandRef(DMDALocalInfo *info, int L, double xi, double eta,
                       const double u[4], const double c[4], const double f[4],
                       PLapCtx *user) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return user->a0 * GradPow(info,du,user->p-2.0,user->eps) * GradInnerProd(info,du,dchiL)
         + (eval(c,xi,eta) * eval(u,xi,eta) - eval(f,xi,eta)) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PLapCtx *user) {
  const double hx = 1.0 / info->mx,  hy = 1.0 / info->my,
               C = 0.25 * hx * hy;
  const int    n = user->quaddegree,
               li[4] = {+1, 0, 0,+1},
               lj[4] = {+1,+1, 0, 0};
  double       x, y;
  int          i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          FF[j][i] = 0.0;
      }
  }

  // loop over all elements, adding element contribution for node we own
  for (j = info->ys-1; j < info->ys+info->ym; j++) {
      y = j * hy;
      for (i = info->xs-1; i < info->xs+info->xm; i++) {
          x = i * hx;   // (x,y) is location of (i,j) node; lower-left corner of element
          const double u[4] = {au[j+1][i+1], au[j+1][i], au[j][i], au[j][i+1]},
                       c[4] = {Ccoeff(x+hx,y+hy,user), Ccoeff(x,   y+hy,user),
                               Ccoeff(x,   y,   user), Ccoeff(x+hx,y,   user)},
                       f[4] = {Frhs(x+hx,y+hy,user), Frhs(x,   y+hy,user),
                               Frhs(x,   y,   user), Frhs(x+hx,y,   user)};
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // do we own node (PP,QQ) ?
              if (PP >= info->xs && PP < info->xs+info->xm
                  && QQ >= info->ys && QQ < info->ys+info->ym) {
                  // loop over quadrature points
                  for (r=0; r<n; r++) {
                      for (s=0; s<n; s++) {
                         FF[QQ][PP] += C * wq[n-1][r] * wq[n-1][s]
                                       * FunIntegrandRef(info,l,zq[n-1][r],zq[n-1][s],
                                                         u,c,f,user);
                      }
                  }
              }
          }
      }
  }
  return 0;
}
//ENDFUNCTION

//STARTMAIN
int main(int argc,char **argv) {
  PetscErrorCode ierr;
  SNES           snes;
  Vec            u,uexact;
  PLapCtx        user;
  DM             da;
  DMDALocalInfo  info;
  double         hx, hy, err;

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = ConfigureCtx(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX,
               3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  hx = 1.0 / info.mx;  hy = 1.0 / info.my;
  ierr = PetscPrintf(PETSC_COMM_WORLD,
            "grid of %d x %d = %d nodes (element dims %gx%g)\n",
            info.mx,info.my,info.mx*info.my,hx,hy); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = VecSet(u,0.0); CHKERRQ(ierr);
//  ierr = getUexact(&info,u,&user); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = getUexact(&info,uexact,&user); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"numerical error:  |u-u_exact|_inf = %.3e\n",
           err); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

