static char help[] = "Solve the p-Laplacian equation in 2D using Q^1 FEM.\n"
"Implements an objective function and a residual (gradient) function, but\n"
"no Jacobian.  Defaults to p=4 and quadrature degree n=2.  Run as one of:\n"
"   ./plap -snes_fd_color             [default]\n"
"   ./plap -snes_mf\n"
"   ./plap -snes_fd                   [does not scale]\n"
"   ./plap -snes_fd_function -snes_fd [does not scale]\n"
"Uses a manufactured solution.\n\n";

#include <petsc.h>

#define COMM PETSC_COMM_WORLD

//STARTCTX
typedef struct {
    DM      da;
    double  p, eps, alpha;
    int     quaddegree;
} PLapCtx;

PetscErrorCode ConfigureCtx(PLapCtx *user) {
    PetscErrorCode ierr;
    user->p = 4.0;
    user->eps = 0.0;
    user->alpha = 1.0;
    user->quaddegree = 2;
    ierr = PetscOptionsBegin(COMM,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                      NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
    if (user->p < 1.0) { SETERRQ(COMM,1,"p >= 1 required"); }
    ierr = PetscOptionsReal("-eps","regularization parameter eps",
                      NULL,user->eps,&(user->eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","parameter alpha in exact solution",
                      NULL,user->alpha,&(user->alpha),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                     NULL,user->quaddegree,&(user->quaddegree),NULL); CHKERRQ(ierr);
    if ((user->quaddegree < 1) || (user->quaddegree > 3)) {
        SETERRQ(COMM,2,"quadrature degree n=1,2,3 only"); }
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}
//ENDCTX

//STARTBDRYINIT
// both the exact solution and the Dirichlet boundary value
double BoundaryG(double x, double y, double alpha) {
    return 0.5 * (x+alpha)*(x+alpha) * (y+alpha)*(y+alpha);
}

// right hand side of PDE
double FRHS(double x, double y, double p, double alpha) {
    const double xs = x + alpha,  ys = y + alpha,  // shifted
                 XX = xs * xs,  YY = ys * ys,  D2 = XX + YY,
                 gamma1 = 1.0 / xs + xs / D2,
                 gamma2 = 1.0 / ys + ys / D2,
                 C = PetscPowScalar(XX * YY * D2, (p - 2.0) / 2.0);
    return - (p - 2.0) * C * (gamma1 * (x + alpha) * YY + gamma2 * XX * (y + alpha))
           - C * D2;
}

PetscErrorCode InitialIterateUExact(DMDALocalInfo *info, Vec u, Vec uexact, PLapCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx-1), hy = 1.0 / (info->my-1);
    double       x, y, **au, **auex;
    int          i,j;
    ierr = DMDAVecGetArray(user->da,u,&au); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,uexact,&auex); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * j;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * i;
            au[j][i] = (1.0 - x) * BoundaryG(0.0,y,user->alpha)
                       + x * BoundaryG(1.0,y,user->alpha);
            auex[j][i] = BoundaryG(x,y,user->alpha); // here: exact soln
        }
    }
    ierr = DMDAVecRestoreArray(user->da,u,&au); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da,uexact,&auex); CHKERRQ(ierr);
    return 0;
}
//ENDBDRYINIT

//STARTFEM
static double xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
              etaL[4] = { 1.0,  1.0, -1.0, -1.0};

double chi(int L, double xi, double eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

typedef struct {
    double  xi, eta;
} gradRef;

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

//STARTTOOLS
double GradInnerProd(DMDALocalInfo *info, gradRef du, gradRef dv) {
    const double hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1),
                 cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

double GradPow(DMDALocalInfo *info, gradRef du, double P, double eps) {
    return PetscPowScalar(GradInnerProd(info,du,du) + eps * eps, P / 2.0);
}

// gets either u(x,y) or g(x,y) for the nodes of the given element (i,j),
// where (i,j) denotes the upper-right node of the element
void GetUorGElement(DMDALocalInfo *info, int i, int j,
                    double **au, double alpha, double *u) {
    const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1),
                 x = i * hx,  y = j * hy;  // (x,y) for node (i,j)
    u[0] = ((i == info->mx-1) || (j == info->my-1))
             ? BoundaryG(x,y,alpha)       : au[j][i];
    u[1] = ((i-1 == 0)        || (j == info->my-1))
             ? BoundaryG(x-hx,y,alpha)    : au[j][i-1];
    u[2] = ((i-1 == 0)        || (j-1 == 0))
             ? BoundaryG(x-hx,y-hy,alpha) : au[j-1][i-1];
    u[3] = ((i == info->mx-1) || (j-1 == 0))
             ? BoundaryG(x,y-hy,alpha)    : au[j-1][i];
}
//ENDTOOLS

#if 0
//STARTOBJECTIVE
double ObjIntegrand(DMDALocalInfo *info, const double f[4], const double u[4],
                    double xi, double eta, double P, double eps) {
    const gradRef du = deval(u,xi,eta);
    return GradPow(info,du,P,eps) / P - eval(f,xi,eta) * eval(u,xi,eta);
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1),
               p = user->p,  eps = user->eps;
  const int    n = user->quaddegree,
               XE = info->xs + info->xm,  YE = info->ys + info->ym;
  double       lobj = 0.0, **af, **ag, f[4], u[4];
  int          i,j,r,s;
  PetscBool    ownele;
  MPI_Comm     com;

  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,user->g,&ag); CHKERRQ(ierr);
  // loop over all elements
  for (j = info->ys; j <= YE; j++) {
      for (i = info->xs; i <= XE; i++) {
          // owned elements include "right" and "top" edges of grid
          ownele = (i < XE || j < YE || i == info->mx || j == info->my);
          if (!ownele) continue;
          f[0] = af[j][i];  f[1] = af[j][i-1];
              f[2] = af[j-1][i-1];  f[3] = af[j-1][i];
          GetUorG(info,i,j,au,ag,u);
          for (r=0; r<n; r++) {
              for (s=0; s<n; s++) {
                  lobj += wq[n-1][r] * wq[n-1][s]
                          * ObjIntegrand(info,f,u,zq[n-1][r],zq[n-1][s],p,eps);
              }
          }
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,user->g,&ag); CHKERRQ(ierr);
  lobj *= 0.25 * hx * hy;

  ierr = PetscObjectGetComm((PetscObject)(info->da),&com); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE
#endif

//STARTFUNCTION
double FunIntegrand(DMDALocalInfo *info, int L,
                    const double f[4], const double u[4],
                    double xi, double eta, double P, double eps) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return GradPow(info,du,P - 2.0,eps) * GradInnerProd(info,du,dchiL)
         - eval(f,xi,eta) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PLapCtx *user) {
  const double hx = 1.0 / (info->mx-1),  hy = 1.0 / (info->my-1),
               p = user->p,  alpha = user->alpha,  eps = user->eps,
               C = 0.25 * hx * hy;
  const int    n = user->quaddegree,
               li[4] = { 0,-1,-1, 0},
               lj[4] = { 0, 0,-1,-1},
               // inclusive ranges of indices for elements which can contribute
               // to the nodal residuals in my processor's patch
               iS = (info->xs < 1) ? 1 : info->xs,
               jS = (info->ys < 1) ? 1 : info->ys,
               iE = (info->xs + info->xm > info->mx-1) ? info->mx-1 : info->xs + info->xm,
               jE = (info->ys + info->ym > info->my-1) ? info->my-1 : info->ys + info->ym;
  double       x, y, f[4], u[4];
  int          i,j,l,r,s,PP,QQ;

  // set boundary residuals and clear other residuals
  for (j = info->ys; j < info->ys + info->ym; j++) {
      y = j * hy;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          x = i * hx;
          // if the node (i,j) is on boundary use g(x,y)
          if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
              FF[j][i] = au[j][i] - BoundaryG(x,y,user->alpha);
          } else {
              FF[j][i] = 0.0;
          }
      }
  }

  // loop over all elements, adding element contribution
  for (j = jS; j <= jE; j++) {  // note upper limit "="
      y = j * hy;
      for (i = iS; i <= iE; i++) {  // note upper limit "="
          x = i * hx;   // (x,y) is location of (i,j) node (upper-right on element)
          f[0] = FRHS(x,   y,   p,alpha);
          f[1] = FRHS(x-hx,y,   p,alpha);
          f[2] = FRHS(x-hx,y-hy,p,alpha);
          f[3] = FRHS(x,   y-hy,p,alpha);
          GetUorGElement(info,i,j,au,alpha,u);
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // add contribution from element only if the node (PP,QQ) is
              // 1. actually an unknown (= interior node) AND
              // 2. we own it
              if (   PP > 0 && PP < info->mx-1 && QQ > 0 && QQ < info->my-1
                  && PP >= info->xs && PP < info->xs+info->xm
                  && QQ >= info->ys && QQ < info->ys+info->ym) {
                  // loop over quadrature points
                  for (r=0; r<n; r++) {
                      for (s=0; s<n; s++) {
                         FF[QQ][PP]
                             += C * wq[n-1][r] * wq[n-1][s]
                                * FunIntegrand(info,l,f,u,zq[n-1][r],zq[n-1][s],
                                               p,eps);
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
  Vec            u, uexact;
  PLapCtx        user;
  DMDALocalInfo  info;
  double         err, hx, hy;

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = ConfigureCtx(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(user.da)); CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
  ierr = DMSetUp(user.da); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  hx = 1.0 / (info.mx-1);  hy = 1.0 / (info.my-1);
  ierr = PetscPrintf(COMM,
            "grid of %d x %d = %d nodes (element dims %gx%g)\n",
            info.mx,info.my,info.mx*info.my,hx,hy); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);

  ierr = InitialIterateUExact(&info,u,uexact,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
//  ierr = DMDASNESSetObjectiveLocal(user.da,
//             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"numerical error:  |u-u_exact|_inf = %.3e\n",
           err); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}
//ENDMAIN

