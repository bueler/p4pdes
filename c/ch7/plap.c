static char help[] = "Solve the p-Laplacian equation in 2D using Q^1 FEM.\n"
"Implements an objective function and a residual (gradient) function, but\n"
"no Jacobian.  Defaults to p=4 and quadrature degree n=2.  Run as one of:\n"
"   ./plap -snes_fd_color             [default]\n"
"   ./plap -snes_mf\n"
"   ./plap -snes_fd                   [does not scale]\n"
"   ./plap -snes_fd_function -snes_fd [does not scale]\n"
"Uses a manufactured solution.\n\n";

#include <petsc.h>
#include "../quadrature.h"

#define COMM PETSC_COMM_WORLD

//STARTCTX
typedef struct {
    double    p, eps, alpha;
    int       quaddegree;
    PetscBool no_residual;
} PLapCtx;
//ENDCTX

PetscErrorCode ConfigureCtx(PLapCtx *user) {
    PetscErrorCode ierr;
    user->p = 4.0;
    user->eps = 0.0;
    user->alpha = 1.0;
    user->quaddegree = 2;
    user->no_residual = PETSC_FALSE;
    ierr = PetscOptionsBegin(COMM,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                      "plap.c",user->p,&(user->p),NULL); CHKERRQ(ierr);
    if (user->p < 1.0) { SETERRQ(COMM,1,"p >= 1 required"); }
    ierr = PetscOptionsReal("-eps","regularization parameter eps",
                      "plap.c",user->eps,&(user->eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","parameter alpha in exact solution",
                      "plap.c",user->alpha,&(user->alpha),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                     "plap.c",user->quaddegree,&(user->quaddegree),NULL); CHKERRQ(ierr);
    if ((user->quaddegree < 1) || (user->quaddegree > 3)) {
        SETERRQ(COMM,2,"quadrature degree n=1,2,3 only"); }
    ierr = PetscOptionsBool("-no_residual","do not set the residual evaluation function",
                      "plap.c",user->no_residual,&(user->no_residual),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}

//STARTEXACT
double Uexact(double x, double y, double alpha) {
    return 0.5 * (x+alpha)*(x+alpha) * (y+alpha)*(y+alpha);
}

double Frhs(double x, double y, PLapCtx *user) {
    const double alf = user->alpha,
                 XX = (x+alf)*(x+alf),
                 YY = (y+alf)*(y+alf),
                 D2 = XX + YY,
                 C = PetscPowScalar(XX * YY * D2, (user->p - 2.0) / 2.0),
                 gamma1 = 1.0/(x+alf) + (x+alf)/D2,
                 gamma2 = 1.0/(y+alf) + (y+alf)/D2;
    return - (user->p - 2.0) * C * (gamma1*(x+alf)*YY + gamma2*XX*(y+alf))
           - C * D2;
}
//ENDEXACT

//STARTINITIALITERATE
PetscErrorCode InitialIterateLocal(DMDALocalInfo *info, Vec u, PLapCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    double       x,y, **au;
    ierr = DMDAVecGetArray(info->da,u,&au); CHKERRQ(ierr);
    for (int j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * (j + 1);
        for (int i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * (i + 1);
            au[j][i] = (1.0 - x) * Uexact(0.0,y,user->alpha)
                       + x * Uexact(1.0,y,user->alpha);
        }
    }
    ierr = DMDAVecRestoreArray(info->da,u,&au); CHKERRQ(ierr);
    return 0;
}
//ENDINITIALITERATE

PetscErrorCode GetUexactLocal(DMDALocalInfo *info, Vec uex, PLapCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    double       x,y, **auex;
    ierr = DMDAVecGetArray(info->da,uex,&auex); CHKERRQ(ierr);
    for (int j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * (j + 1);
        for (int i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * (i + 1);
            auex[j][i] = Uexact(x,y,user->alpha);
        }
    }
    ierr = DMDAVecRestoreArray(info->da,uex,&auex); CHKERRQ(ierr);
    return 0;
}

//STARTFEM
static double xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
              etaL[4] = { 1.0,  1.0, -1.0, -1.0};

double chi(int L, double xi, double eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

// evaluate v(xi,eta) on reference element using local node numbering
double eval(const double v[4], double xi, double eta) {
    double sum = 0.0;
    for (int L=0; L<4; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}
//ENDFEM

//STARTGRADFEM
typedef struct {
    double  xi, eta;
} gradRef;

gradRef dchi(int L, double xi, double eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

// evaluate partial derivs of v(xi,eta) on reference element
gradRef deval(const double v[4], double xi, double eta) {
    gradRef sum = {0.0,0.0}, tmp;
    for (int L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}
//ENDGRADFEM

//STARTTOOLS
void GetUorG(DMDALocalInfo *info, int i, int j, double **au, double *u,
             PLapCtx *user) {
    const double hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1),
                 x = hx * (i + 1),  y = hy * (j + 1);
    u[0] = ((i == info->mx) || (j == info->my))
             ? Uexact(x,   y,   user->alpha) : au[j][i];
    u[1] = ((i == 0)  || (j == info->my))
             ? Uexact(x-hx,y,   user->alpha) : au[j][i-1];
    u[2] = ((i == 0)  || (j == 0))
             ? Uexact(x-hx,y-hy,user->alpha) : au[j-1][i-1];
    u[3] = ((i == info->mx) || (j == 0))
             ? Uexact(x,   y-hy,user->alpha) : au[j-1][i];
}

double GradInnerProd(DMDALocalInfo *info, gradRef du, gradRef dv) {
    const double hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1),
                 cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

double GradPow(DMDALocalInfo *info, gradRef du, double P, double eps) {
    return PetscPowScalar(GradInnerProd(info,du,du) + eps * eps, P / 2.0);
}
//ENDTOOLS

//STARTOBJECTIVE
double ObjIntegrandRef(DMDALocalInfo *info, const double f[4], const double u[4],
                       double xi, double eta, double p, double eps) {
    const gradRef du = deval(u,xi,eta);
    return GradPow(info,du,p,eps) / p - eval(f,xi,eta) * eval(u,xi,eta);
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1);
  const Quad1D q = gausslegendre[user->quaddegree-1];
  const int    XE = info->xs + info->xm,  YE = info->ys + info->ym;
  double       x, y, lobj = 0.0, u[4];
  int          i,j,r,s;
  MPI_Comm     com;

  // loop over all elements
  for (j = info->ys; j <= YE; j++) {
      y = hy * (j + 1);
      for (i = info->xs; i <= XE; i++) {
          x = hx * (i + 1);
          // owned elements include "right" and "top" edges of grid
          if (i < XE || j < YE || i == info->mx || j == info->my) {
              const double f[4] = {Frhs(x,   y,   user),
                                   Frhs(x-hx,y,   user),
                                   Frhs(x-hx,y-hy,user),
                                   Frhs(x,   y-hy,user)};
              GetUorG(info,i,j,au,u,user);
              // loop over quadrature points
              for (r = 0; r < q.n; r++) {
                  for (s = 0; s < q.n; s++) {
                      lobj += q.w[r] * q.w[s]
                              * ObjIntegrandRef(info,f,u,q.xi[r],q.xi[s],
                                                user->p,user->eps);
                  }
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
double FunIntegrandRef(DMDALocalInfo *info, int L,
                       const double f[4], const double u[4],
                       double xi, double eta, double p, double eps) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return GradPow(info,du,p - 2.0,eps) * GradInnerProd(info,du,dchiL)
         - eval(f,xi,eta) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PLapCtx *user) {
  const double hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1);
  const Quad1D q = gausslegendre[user->quaddegree-1];
  const int    XE = info->xs + info->xm,  YE = info->ys + info->ym,
               li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
  double       x, y, u[4];
  int          i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < YE; j++)
      for (i = info->xs; i < XE; i++)
          FF[j][i] = 0.0;

  // loop over all elements
  for (j = info->ys; j <= YE; j++) {
      y = hy * (j + 1);
      for (i = info->xs; i <= XE; i++) {
          x = hx * (i + 1);
          const double f[4] = {Frhs(x,   y,   user),
                               Frhs(x-hx,y,   user),
                               Frhs(x-hx,y-hy,user),
                               Frhs(x,   y-hy,user)};
          GetUorG(info,i,j,au,u,user);
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // only update residual if we own node
              if (PP >= info->xs && PP < XE && QQ >= info->ys && QQ < YE) {
                  // loop over quadrature points
                  for (r = 0; r < q.n; r++) {
                      for (s = 0; s < q.n; s++) {
                         FF[QQ][PP]
                             += 0.25 * hx * hy * q.w[r] * q.w[s]
                                * FunIntegrandRef(info,l,f,u,q.xi[r],q.xi[s],
                                                  user->p,user->eps);
                      }
                  }
              }
          }
      }
  }
  return 0;
}
//ENDFUNCTION

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  Vec            u, uexact;
  PLapCtx        user;
  DMDALocalInfo  info;
  double         err;

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = ConfigureCtx(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
            "grid of %d x %d = %d interior nodes\n",
            info.mx,info.my,info.mx*info.my); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  if (!user.no_residual) {
      ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                 (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = InitialIterateLocal(&info,u,&user); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = GetUexactLocal(&info,uexact,&user); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"numerical error:  |u-u_exact|_inf = %.3e\n",
           err); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

