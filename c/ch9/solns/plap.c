static char help[] = "Solve the p-Laplacian equation in 2D using Q_1 FEM.\n"
"Implements an objective function and a residual (gradient) function, but\n"
"no Jacobian.  Defaults to p=4 and quadrature degree n=2.  Run as one of:\n"
"   ./plap -snes_fd_color             [default]\n"
"   ./plap -snes_mf\n"
"   ./plap -snes_fd                   [does not scale]\n"
"   ./plap -snes_fd_function -snes_fd [does not scale]\n"
"Uses a manufactured solution.\n"
"This is NOT a recommended example for further work because of the weird way\n"
"it handles the Dirichlet boundary values.\n\n";

#include <petsc.h>
#include "../../interlude/quadrature.h"

/* cool dendritic failure:
    ./plap -snes_fd_color -snes_converged_reason -ksp_converged_reason -pc_type mg -plap_p 10.0 -da_refine 6 -snes_monitor_solution draw
succeeds in 11 snes iterations (on fine grid) with grid sequencing:
    ./plap -snes_fd_color -snes_converged_reason -ksp_converged_reason -pc_type mg -plap_p 10.0 -snes_grid_sequence 6 -snes_monitor_solution draw
*/

typedef struct {
    PetscReal  p, eps, alpha;
    PetscInt   quaddegree;
    PetscBool  no_residual;
} PLapCtx;

PetscErrorCode ConfigureCtx(PLapCtx *user) {
    user->p = 4.0;
    user->eps = 0.0;
    user->alpha = 1.0;
    user->quaddegree = 2;
    user->no_residual = PETSC_FALSE;
    PetscOptionsBegin(PETSC_COMM_WORLD,"plap_","p-laplacian solver options","");
    PetscCall(PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                      "plap.c",user->p,&(user->p),NULL));
    if (user->p < 1.0) { SETERRQ(PETSC_COMM_WORLD,1,"p >= 1 required"); }
    PetscCall(PetscOptionsReal("-eps","regularization parameter eps",
                      "plap.c",user->eps,&(user->eps),NULL));
    PetscCall(PetscOptionsReal("-alpha","parameter alpha in exact solution",
                      "plap.c",user->alpha,&(user->alpha),NULL));
    PetscCall(PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                     "plap.c",user->quaddegree,&(user->quaddegree),NULL));
    if ((user->quaddegree < 1) || (user->quaddegree > 3)) {
        SETERRQ(PETSC_COMM_WORLD,2,"quadrature degree n=1,2,3 only"); }
    PetscCall(PetscOptionsBool("-no_residual","do not set the residual evaluation function",
                      "plap.c",user->no_residual,&(user->no_residual),NULL));
    PetscOptionsEnd();
    return 0;
}

PetscReal Uexact(PetscReal x, PetscReal y, PetscReal alpha) {
    return 0.5 * (x+alpha)*(x+alpha) * (y+alpha)*(y+alpha);
}

PetscReal Frhs(PetscReal x, PetscReal y, PLapCtx *user) {
    const PetscReal alf = user->alpha,
                    XX = (x+alf)*(x+alf),
                    YY = (y+alf)*(y+alf),
                    D2 = XX + YY,
                    C = PetscPowScalar(XX * YY * D2, (user->p - 2.0) / 2.0),
                    gamma1 = 1.0/(x+alf) + (x+alf)/D2,
                    gamma2 = 1.0/(y+alf) + (y+alf)/D2;
    return - (user->p - 2.0) * C * (gamma1*(x+alf)*YY + gamma2*XX*(y+alf))
           - C * D2;
}

PetscErrorCode InitialIterateLocal(DMDALocalInfo *info, Vec u, PLapCtx *user) {
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    PetscReal       x, y, **au;
    PetscInt        i, j;
    PetscCall(DMDAVecGetArray(info->da,u,&au));
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * (j + 1);
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * (i + 1);
            au[j][i] = (1.0 - x) * Uexact(0.0,y,user->alpha)
                       + x * Uexact(1.0,y,user->alpha);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da,u,&au));
    return 0;
}

PetscErrorCode GetUexactLocal(DMDALocalInfo *info, Vec uex, PLapCtx *user) {
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    PetscReal       x, y, **auex;
    PetscInt        i, j;
    PetscCall(DMDAVecGetArray(info->da,uex,&auex));
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * (j + 1);
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * (i + 1);
            auex[j][i] = Uexact(x,y,user->alpha);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da,uex,&auex));
    return 0;
}

PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
          etaL[4] = { 1.0,  1.0, -1.0, -1.0};

PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

// evaluate v(xi,eta) on reference element using local node numbering
PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    PetscReal sum = 0.0;
    PetscInt  L;
    for (L=0; L<4; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

typedef struct {
    PetscReal  xi, eta;
} gradRef;

gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

// evaluate partial derivs of v(xi,eta) on reference element
gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

void GetUorG(DMDALocalInfo *info, PetscInt i, PetscInt j, PetscReal **au, PetscReal *u,
             PLapCtx *user) {
    const PetscReal hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1),
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

PetscReal GradInnerProd(DMDALocalInfo *info, gradRef du, gradRef dv) {
    const PetscReal hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1),
                    cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

PetscReal GradPow(DMDALocalInfo *info, gradRef du, PetscReal P, PetscReal eps) {
    return PetscPowScalar(GradInnerProd(info,du,du) + eps * eps, P / 2.0);
}

PetscReal ObjIntegrandRef(DMDALocalInfo *info,
                          const PetscReal f[4], const PetscReal u[4],
                          PetscReal xi, PetscReal eta, PetscReal p, PetscReal eps) {
    const gradRef du = deval(u,xi,eta);
    return GradPow(info,du,p,eps) / p - eval(f,xi,eta) * eval(u,xi,eta);
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  const PetscReal hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1);
  const Quad1D    q = gausslegendre[user->quaddegree-1];
  const PetscInt  XE = info->xs + info->xm,  YE = info->ys + info->ym;
  PetscReal       x, y, lobj = 0.0, u[4];
  PetscInt        i,j,r,s;
  MPI_Comm        com;

  // loop over all elements
  for (j = info->ys; j <= YE; j++) {
      y = hy * (j + 1);
      for (i = info->xs; i <= XE; i++) {
          x = hx * (i + 1);
          // owned elements include "right" and "top" edges of grid
          if (i < XE || j < YE || i == info->mx || j == info->my) {
              const PetscReal f[4] = {Frhs(x,   y,   user),
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

  PetscCall(PetscObjectGetComm((PetscObject)(info->da),&com));
  PetscCall(MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,com));
  return 0;
}

PetscReal FunIntegrandRef(DMDALocalInfo *info, PetscInt L,
                          const PetscReal f[4], const PetscReal u[4],
                          PetscReal xi, PetscReal eta, PetscReal p, PetscReal eps) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return GradPow(info,du,p - 2.0,eps) * GradInnerProd(info,du,dchiL)
         - eval(f,xi,eta) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PLapCtx *user) {
  const PetscReal hx = 1.0 / (info->mx+1),  hy = 1.0 / (info->my+1);
  const Quad1D    q = gausslegendre[user->quaddegree-1];
  const PetscInt  XE = info->xs + info->xm,  YE = info->ys + info->ym,
                  li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
  PetscReal       x, y, u[4];
  PetscInt        i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < YE; j++)
      for (i = info->xs; i < XE; i++)
          FF[j][i] = 0.0;

  // loop over all elements
  for (j = info->ys; j <= YE; j++) {
      y = hy * (j + 1);
      for (i = info->xs; i <= XE; i++) {
          x = hx * (i + 1);
          const PetscReal f[4] = {Frhs(x,   y,   user),
                                  Frhs(x-hx,y,   user),
                                  Frhs(x-hx,y-hy,user),
                                  Frhs(x,   y-hy,user)};
          GetUorG(info,i,j,au,u,user);
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // only update residual if we own node
              if (PP >= info->xs && PP < XE
                  && QQ >= info->ys && QQ < YE) {
                  // loop over quadrature points
                  for (r = 0; r < q.n; r++) {
                      for (s = 0; s < q.n; s++) {
                         FF[QQ][PP]
                             += 0.25 * hx * hy * q.w[r] * q.w[s]
                                * FunIntegrandRef(info,l,f,u,
                                                  q.xi[r],q.xi[s],
                                                  user->p,user->eps);
                      }
                  }
              }
          }
      }
  }
  return 0;
}

int main(int argc,char **argv) {
  DM             da, da_after;
  SNES           snes;
  Vec            u_initial, u, u_exact;
  PLapCtx        user;
  DMDALocalInfo  info;
  PetscReal      err, hx, hy;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(ConfigureCtx(&user));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(DMDAGetLocalInfo(da,&info));
  hx = 1.0 / (info.mx+1);
  hy = 1.0 / (info.my+1);
  PetscCall(DMDASetUniformCoordinates(da,hx,1.0-hx,hy,1.0-hy,0.0,1.0));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user));
  if (!user.no_residual) {
      PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
                 (DMDASNESFunction)FormFunctionLocal,&user));
  }
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(da,&u_initial));
  PetscCall(InitialIterateLocal(&info,u_initial,&user));
  PetscCall(SNESSolve(snes,NULL,u_initial));
  PetscCall(VecDestroy(&u_initial));
  PetscCall(DMDestroy(&da));

  PetscCall(SNESGetSolution(snes,&u));
  PetscCall(VecDuplicate(u,&u_exact));
  PetscCall(SNESGetDM(snes,&da_after));
  PetscCall(DMDAGetLocalInfo(da_after,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
            "grid of %d x %d = %d interior nodes\n",
            info.mx,info.my,info.mx*info.my));
  PetscCall(GetUexactLocal(&info,u_exact,&user));
  PetscCall(VecAXPY(u,-1.0,u_exact));    // u <- u + (-1.0) uexact
  PetscCall(VecNorm(u,NORM_INFINITY,&err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"numerical error:  |u-u_exact|_inf = %.3e\n",
           err));

  VecDestroy(&u_exact);  SNESDestroy(&snes);
  PetscCall(PetscFinalize());
  return 0;
}
