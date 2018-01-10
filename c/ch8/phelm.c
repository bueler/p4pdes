static char help[] =
"Solves the p-Helmholtz equation in 2D using Q^1 FEM.  Option prefix -ph_.\n"
"Implements an objective function and a residual (gradient) function, but\n"
"no Jacobian.  Defaults to p=4 and quadrature degree n=2.  Can run with\n"
"only an objective function; use -ph_no_residual -snes_fd_function.\n"
"Exact (manufactured) solution available in cases p=2,4.\n\n";

#include <petsc.h>
#include "../quadrature.h"

typedef struct {
    double     p, eps, a, b;
    int        quaddegree;
} PHelmCtx;

extern PetscErrorCode GetUExactLocal(DMDALocalInfo*, Vec, PHelmCtx*);
extern PetscErrorCode FormObjectiveLocal(DMDALocalInfo*, double**, double*, PHelmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double**, double**, PHelmCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u_initial, u;
    PHelmCtx       user;
    DMDALocalInfo  info;
    PetscBool      no_objective = PETSC_FALSE,
                   no_residual = PETSC_FALSE,
                   exact_init = PETSC_FALSE;
    int            tmpa = 1, tmpb = 0;

    PetscInitialize(&argc,&argv,NULL,help);
    user.p = 4.0;
    user.eps = 0.0;
    user.quaddegree = 2;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ph_","p-Helmholtz solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","regularization parameter eps",
                  "plap.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exact_init","use exact solution to initialize (p=2,4 only)",
                  "plap.c",exact_init,&(exact_init),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_objective","do not set the objective evaluation function",
                  "plap.c",no_objective,&(no_objective),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_residual","do not set the residual evaluation function",
                  "plap.c",no_residual,&(no_residual),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                  "plap.c",user.p,&(user.p),NULL); CHKERRQ(ierr);
    if (user.p < 1.0) {
         SETERRQ(PETSC_COMM_WORLD,1,"p >= 1 required");
    }
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                 "plap.c",user.quaddegree,&(user.quaddegree),NULL); CHKERRQ(ierr);
    if ((user.quaddegree < 1) || (user.quaddegree > 3)) {
        SETERRQ(PETSC_COMM_WORLD,2,"quadrature degree n=1,2,3 only");
    }
    ierr = PetscOptionsInt("-soln_a","integer parameter a in exact solution (p=2,4 only)",
                  "plap.c",tmpa,&tmpa,NULL); CHKERRQ(ierr);
    user.a = (double)tmpa;
    ierr = PetscOptionsInt("-soln_b","integer parameter b in exact solution (p=2,4 only)",
                  "plap.c",tmpb,&tmpb,NULL); CHKERRQ(ierr);
    user.b = (double)tmpb;
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
           2,2,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    if (!no_objective) {
        ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
    }
    if (no_residual) {
        // why isn't this the default?  why no programmatic way to set?
        ierr = PetscOptionsSetValue(NULL,"-snes_fd_function_eps","0.0"); CHKERRQ(ierr);
    } else {
        ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    // set initial iterate
    ierr = DMCreateGlobalVector(da,&u_initial);CHKERRQ(ierr);
    if (exact_init) {
        ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
        ierr = GetUExactLocal(&info,u_initial,&user); CHKERRQ(ierr);
    } else {
        ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    }

    // solve and clean up
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid",info.mx,info.my); CHKERRQ(ierr);

    // evaluate numerical error if available
    if (user.p == 2.0 || user.p == 4.0) {
        Vec     u_exact;
        double  err;
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = GetUExactLocal(&info,u_exact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
        ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"; numerical error:  |u-u_exact|_inf = %.3e\n",
               err); CHKERRQ(ierr);
        ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"...\n"); CHKERRQ(ierr);
    }

    SNESDestroy(&snes);
    return PetscFinalize();
}

static double UExact(double x, double y, PHelmCtx *user) {
    return cos(user->a * PETSC_PI * x) * cos(user->b * PETSC_PI * y);
}

static double Frhs(double x, double y, PHelmCtx *user) {
    if (user->p == 2.0) {
        const double u = UExact(x,y,user),
                     api = user->a * PETSC_PI,
                     bpi = user->b * PETSC_PI;
        return (api * api + bpi * bpi + 1.0) * u;
    } else if (user->p == 4.0) {
        const double u = UExact(x,y,user),
                     api = user->a * PETSC_PI,
                     bpi = user->b * PETSC_PI,
                     api2 = api * api,
                     bpi2 = bpi * bpi,
                     lapu = - (api2 + bpi2) * u,
                     sax = sin(api * x),
                     cax = cos(api * x),
                     sby = sin(bpi * y),
                     cby = cos(bpi * y),
                     ux = - api * sax * cby,
                     uy = - bpi * cax * sby,
                     w = ux * ux + uy * uy,
                     wx = api * sin(2 * api * x) * (api2 * cby * cby - bpi2 * sby * sby),
                     wy = bpi * sin(2 * bpi * y) * (bpi2 * cax * cax - api2 * sax * sax);
        return - wx * ux - wy * uy - w * lapu + u;
    } else {
        return 1.0;
    }
}

PetscErrorCode GetUExactLocal(DMDALocalInfo *info, Vec uex, PHelmCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx - 1), hy = 1.0 / (info->my - 1);
    double       x, y, **auex;
    int          i, j;
    ierr = DMDAVecGetArray(info->da,uex,&auex); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            auex[j][i] = UExact(x,y,user);
        }
    }
    ierr = DMDAVecRestoreArray(info->da,uex,&auex); CHKERRQ(ierr);
    return 0;
}

//STARTFEM
static double xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
              etaL[4] = { 1.0,  1.0, -1.0, -1.0};

static double chi(int L, double xi, double eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

// evaluate v(xi,eta) on reference element using local node numbering
static double eval(const double v[4], double xi, double eta) {
    double sum = 0.0;
    int    L;
    for (L=0; L<4; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

typedef struct {
    double  xi, eta;
} gradRef;

static gradRef dchi(int L, double xi, double eta) {
    const gradRef result = {0.25 * xiL[L]  * (1.0 + etaL[L] * eta),
                            0.25 * etaL[L] * (1.0 + xiL[L]  * xi)};
    return result;
}

// evaluate partial derivs of v(xi,eta) on reference element
static gradRef deval(const double v[4], double xi, double eta) {
    gradRef sum = {0.0,0.0}, tmp;
    int     L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi += v[L] * tmp.xi;  sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

static double GradInnerProd(double hx, double hy, gradRef du, gradRef dv) {
    const double cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

static double GradPow(double hx, double hy, gradRef du, double P, double eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps * eps, P / 2.0);
}
//ENDFEM

//STARTOBJECTIVE
static double ObjIntegrandRef(DMDALocalInfo *info,
                       const double f[4], const double u[4],
                       double xi, double eta, PHelmCtx *user) {
    const gradRef du = deval(u,xi,eta);
    const double  hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1),
                  uu = eval(u,xi,eta);
    return GradPow(hx,hy,du,user->p,user->eps) / user->p + 0.5 * uu * uu - eval(f,xi,eta) * uu;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PHelmCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  const Quad1D q = gausslegendre[user->quaddegree-1];
  double       x, y, lobj = 0.0;
  int          i,j,r,s;
  MPI_Comm     com;

  //ierr = PetscPrintf(PETSC_COMM_WORLD,"in FormObjectiveLocal():\n"); CHKERRQ(ierr);
  // loop over all elements
  for (j = info->ys; j < info->ys + info->ym; j++) {
      if (j == 0)
          continue;
      y = j * hy;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if (i == 0)
              continue;
          x = i * hx;
          //ierr = PetscPrintf(PETSC_COMM_WORLD,"    element i,j=%d,%d\n",i,j); CHKERRQ(ierr);
          const double f[4] = {Frhs(x,   y,   user),
                               Frhs(x-hx,y,   user),
                               Frhs(x-hx,y-hy,user),
                               Frhs(x,   y-hy,user)};
          const double u[4] = {au[j][i],
                               au[j][i-1],
                               au[j-1][i-1],
                               au[j-1][i]};
          // loop over quadrature points on this element
          for (r = 0; r < q.n; r++) {
              for (s = 0; s < q.n; s++) {
                  lobj += q.w[r] * q.w[s]
                          * ObjIntegrandRef(info,f,u,q.xi[r],q.xi[s],user);
              }
          }
      }
  }
  lobj *= hx * hy / 4.0;  // from change of variables formula
  ierr = PetscObjectGetComm((PetscObject)(info->da),&com); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
static double FunIntegrandRef(DMDALocalInfo *info, int L,
                       const double f[4], const double u[4],
                       double xi, double eta, PHelmCtx *user) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  const double  hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  return GradPow(hx,hy,du,user->p - 2.0,user->eps) * GradInnerProd(hx,hy,du,dchiL)
         + (eval(u,xi,eta) - eval(f,xi,eta)) * chi(L,xi,eta);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PHelmCtx *user) {
  const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  const Quad1D q = gausslegendre[user->quaddegree-1];
  const int    li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
  double       x, y;
  int          i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < info->ys + info->ym; j++)
      for (i = info->xs; i < info->xs + info->xm; i++)
          FF[j][i] = 0.0;

  // loop over all elements
  for (j = info->ys; j < info->ys + info->ym; j++) {
      if (j == 0)
          continue;
      y = j * hy;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if (i == 0)
              continue;
          x = i * hx;
          const double f[4] = {Frhs(x,   y,   user),
                               Frhs(x-hx,y,   user),
                               Frhs(x-hx,y-hy,user),
                               Frhs(x,   y-hy,user)};
          const double u[4] = {au[j][i],
                               au[j][i-1],
                               au[j-1][i-1],
                               au[j-1][i]};
          // loop over corners of element i,j
          for (l = 0; l < 4; l++) {
              PP = i + li[l];
              QQ = j + lj[l];
              // only update residual if we own node
              if (PP >= info->xs && PP < info->xs + info->xm
                  && QQ >= info->ys && QQ < info->ys + info->ym) {
                  // loop over quadrature points
                  for (r = 0; r < q.n; r++) {
                      for (s = 0; s < q.n; s++) {
                         FF[QQ][PP]
                             += 0.25 * hx * hy * q.w[r] * q.w[s]
                                * FunIntegrandRef(info,l,f,u,
                                                  q.xi[r],q.xi[s],user);
                      }
                  }
              }
          }
      }
  }
  return 0;
}
//ENDFUNCTION

