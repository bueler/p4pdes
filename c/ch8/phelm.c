static char help[] =
"Solves the p-Helmholtz equation in 2D using Q^1 FEM.  Option prefix -ph_.\n"
"Equation is\n"
"    - div( |grad u|^{p-2} grad u ) + u = f\n"
"with homogeneous Neumann boundary conditions.  The objective functional is\n"
"    I[u] = int_Omega (1/p) |grad u|^p + (1/2) u^2 - f u.\n"
"Implements objective and residual (gradient) but no Jacobian.  Covers cases\n"
"1 <= p <= 2.  Defaults to easy linear problem with p=2.  Defaults to\n"
"quadrature degree 2.  Can run with only an objective function; use\n"
"-ph_no_residual -snes_fd_function.\n\n";

#include <petsc.h>
#include "../quadrature.h"

typedef struct {
    double     p, eps;
    int        quaddegree;
    double     (*f)(double x, double y, double p, double eps);
} PHelmCtx;

static double f_constant(double x, double y, double p, double eps) {
    return 1.0;
}

static double u_exact_cosines(double x, double y) {
    return cos(PETSC_PI * x) * cos(PETSC_PI * y);
}

static double f_cosines(double x, double y, double p, double eps) {
    const double uu = u_exact_cosines(x,y),
                 pi2 = PETSC_PI * PETSC_PI,
                 lapu = - 2 * pi2 * uu;
    if (p == 2.0) {
        return - lapu + uu;
    } else {
        const double
            ux = - PETSC_PI * sin(PETSC_PI * x) * cos(PETSC_PI * y),
            uy = - PETSC_PI * cos(PETSC_PI * x) * sin(PETSC_PI * y),
            w = ux * ux + uy * uy + eps * eps,  // FIXME consider capping |f| instead
            pi3 = pi2 * PETSC_PI,
            wx = pi3 * sin(2 * PETSC_PI * x) * cos(2 * PETSC_PI * y),
            wy = pi3 * cos(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y);
        const double s = (p - 2) / 2;  //  -1/2 <= s <= 0
        return - s * PetscPowScalar(w,s-1) * (wx * ux + wy * uy)
               - PetscPowScalar(w,s) * lapu
               + uu;
    }
}

typedef enum {CONSTANT, COSINES} ProblemType;
static const char* ProblemTypes[] = {"constant","cosines",
                                     "ProblemType", "", NULL};

extern PetscErrorCode GetUExactCosinesLocal(DMDALocalInfo*, Vec);
extern PetscErrorCode FormObjectiveLocal(DMDALocalInfo*, double**, double*, PHelmCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double**, double**, PHelmCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    PHelmCtx       user;
    DMDALocalInfo  info;
    ProblemType    problem = COSINES;
    PetscBool      no_objective = PETSC_FALSE,
                   no_residual = PETSC_FALSE,
                   exact_init = PETSC_FALSE;
    double         err;

    PetscInitialize(&argc,&argv,NULL,help);
    user.p = 2.0;
    user.eps = 0.0;
    user.quaddegree = 2;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ph_","p-Helmholtz solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","regularization parameter eps",
                  "plap.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exact_init","use exact solution to initialize",
                  "plap.c",exact_init,&(exact_init),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_objective","do not set the objective evaluation function",
                  "plap.c",no_objective,&(no_objective),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_residual","do not set the residual evaluation function",
                  "plap.c",no_residual,&(no_residual),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p <= 2",
                  "plap.c",user.p,&(user.p),NULL); CHKERRQ(ierr);
    if (user.p < 1.0) {
         SETERRQ(PETSC_COMM_WORLD,1,"p >= 1 required");
    }
    if (user.p > 2.0) {
         SETERRQ(PETSC_COMM_WORLD,2,"p <= 2 required");
    }
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                 "plap.c",user.quaddegree,&(user.quaddegree),NULL); CHKERRQ(ierr);
    if ((user.quaddegree < 1) || (user.quaddegree > 3)) {
        SETERRQ(PETSC_COMM_WORLD,3,"quadrature degree n=1,2,3 only");
    }
    ierr = PetscOptionsEnum("-problem","problem type determines right side f(x,y)",
                 "plap.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,
                 NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
           2,2,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

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

    // set initial iterate and right-hand side
    ierr = DMCreateGlobalVector(da,&u_initial);CHKERRQ(ierr);
    ierr = VecSet(u_initial,0.5); CHKERRQ(ierr);
    switch (problem) {
        case CONSTANT:
            if (exact_init) {
                ierr = VecSet(u_initial,1.0); CHKERRQ(ierr);
            }
            user.f = &f_constant;
            break;
        case COSINES:
            if (exact_init) {
                ierr = GetUExactCosinesLocal(&info,u_initial); CHKERRQ(ierr);
            }
            user.f = &f_cosines;
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,4,"unknown problem type\n");
    }

    // solve and clean up
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

    // evaluate numerical error
    ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
    switch (problem) {
        case CONSTANT:
            ierr = VecSet(u_exact,1.0); CHKERRQ(ierr);
            break;
        case COSINES:
            ierr = GetUExactCosinesLocal(&info,u_exact); CHKERRQ(ierr);
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,5,"unknown problem type\n");
    }
    ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid; p=%.3f; numerical error:  |u-u_exact|_inf = %.3e\n",
        info.mx,info.my,user.p,err); CHKERRQ(ierr);
    VecDestroy(&u_exact);  SNESDestroy(&snes);
    return PetscFinalize();
}

PetscErrorCode GetUExactCosinesLocal(DMDALocalInfo *info, Vec uex) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx - 1), hy = 1.0 / (info->my - 1);
    double       x, y, **auex;
    int          i, j;
    ierr = DMDAVecGetArray(info->da,uex,&auex); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            auex[j][i] = u_exact_cosines(x,y);
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

static double GradInnerProd(double hx, double hy,
                            gradRef du, gradRef dv) {
    const double cx = 4.0 / (hx * hx),  cy = 4.0 / (hy * hy);
    return cx * du.xi  * dv.xi + cy * du.eta * dv.eta;
}

static double GradPow(double hx, double hy,
                      gradRef du, double P, double eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0);
}
//ENDFEM

//STARTOBJECTIVE
static double ObjIntegrandRef(DMDALocalInfo *info,
                       const double ff[4], const double uu[4],
                       double xi, double eta, PHelmCtx *user) {
    const gradRef du = deval(uu,xi,eta);
    const double  hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1),
                  u = eval(uu,xi,eta);
    return GradPow(hx,hy,du,user->p,user->eps) / user->p + 0.5 * u * u
           - eval(ff,xi,eta) * u;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PHelmCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  const Quad1D q = gausslegendre[user->quaddegree-1];
  double       x, y, lobj = 0.0;
  int          i,j,r,s;
  MPI_Comm     com;

  // loop over all elements
  for (j = info->ys; j < info->ys + info->ym; j++) {
      if (j == 0)
          continue;
      y = j * hy;
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if (i == 0)
              continue;
          x = i * hx;
          const double ff[4] = {user->f(x,y,user->p,user->eps),
                                user->f(x-hx,y,user->p,user->eps),
                                user->f(x-hx,y-hy,user->p,user->eps),
                                user->f(x,y-hy,user->p,user->eps)};
          const double uu[4] = {au[j][i],au[j][i-1],
                                au[j-1][i-1],au[j-1][i]};
          // loop over quadrature points on this element
          for (r = 0; r < q.n; r++) {
              for (s = 0; s < q.n; s++) {
                  lobj += q.w[r] * q.w[s]
                        * ObjIntegrandRef(info,ff,uu,q.xi[r],q.xi[s],user);
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
                       const double ff[4], const double uu[4],
                       double xi, double eta, PHelmCtx *user) {
  const gradRef du    = deval(uu,xi,eta),
                dchiL = dchi(L,xi,eta);
  const double  hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  return GradPow(hx,hy,du,user->p - 2.0,user->eps)
           * GradInnerProd(hx,hy,du,dchiL)
         + (eval(uu,xi,eta) - eval(ff,xi,eta)) * chi(L,xi,eta);
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
          const double ff[4] = {user->f(x,y,user->p,user->eps),
                                user->f(x-hx,y,user->p,user->eps),
                                user->f(x-hx,y-hy,user->p,user->eps),
                                user->f(x,y-hy,user->p,user->eps)};
          const double uu[4] = {au[j][i],au[j][i-1],
                                au[j-1][i-1],au[j-1][i]};
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
                                * FunIntegrandRef(info,l,ff,uu,
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

