static char help[] =
"Solves the p-Helmholtz equation in 2D using Q^1 FEM.  Option prefix -ph_.\n"
"Problem is posed as minimizing this objective functional over W^{1,p}:\n"
"    I[u] = int_Omega (1/p) |grad u|^p + (1/2) u^2 - f u.\n"
"The strong form equation, namely setting the gradient to zero, is a PDE\n"
"    - div( |grad u|^{p-2} grad u ) + u = f\n"
"subject to homogeneous Neumann boundary conditions.  Implements objective\n"
"and gradient (residual) but no Hessian (Jacobian).  Covers cases 1 <= p.\n"
"Defaults to easy linear problem with p=2 and quadrature degree 2.\n"
"Can be run with only an objective function; use -ph_no_gradient\n"
"-snes_fd_function.\n\n";

#include <petsc.h>
#include "../quadrature.h"

typedef struct {
    double     p, eps;
    int        quadpts;
    double     (*f)(double x, double y, double p, double eps);
} PHelmCtx;

static double f_constant(double x, double y, double p, double eps) {
    return 1.0;
}

static double u_exact_cosines(double x, double y, double p, double eps) {
    return cos(PETSC_PI * x) * cos(PETSC_PI * y);
}

static double f_cosines(double x, double y, double p, double eps) {
    const double uu = u_exact_cosines(x,y,p,eps),
                 pi2 = PETSC_PI * PETSC_PI,
                 lapu = - 2 * pi2 * uu;
    if (p == 2.0) {
        return - lapu + uu;
    } else {
        const double
            ux = - PETSC_PI * sin(PETSC_PI * x) * cos(PETSC_PI * y),
            uy = - PETSC_PI * cos(PETSC_PI * x) * sin(PETSC_PI * y),
            // note regularization changes f(x,y) but not u(x,y):
            w = ux * ux + uy * uy + eps * eps,
            pi3 = pi2 * PETSC_PI,
            wx = pi3 * sin(2 * PETSC_PI * x) * cos(2 * PETSC_PI * y),
            wy = pi3 * cos(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y);
        const double s = (p - 2) / 2;  //  -1/2 <= s <= 0
        return - s * PetscPowScalar(w,s-1) * (wx * ux + wy * uy)
               - PetscPowScalar(w,s) * lapu + uu;
    }
}

typedef enum {CONSTANT, COSINES} ProblemType;
static const char* ProblemTypes[] = {"constant","cosines",
                                     "ProblemType", "", NULL};

extern PetscErrorCode GetVecFromFunction(DMDALocalInfo*, Vec,
                         double (*)(double, double, double, double), PHelmCtx*);
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
                   no_gradient = PETSC_FALSE,
                   exact_init = PETSC_FALSE,
                   view_f = PETSC_FALSE;
    double         err;

    PetscInitialize(&argc,&argv,NULL,help);
    user.p = 2.0;
    user.eps = 0.0;
    user.quadpts = 2;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ph_",
                  "p-Helmholtz solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps",
                  "regularization parameter eps",
                  "phelm.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exact_init",
                  "use exact solution to initialize",
                  "phelm.c",exact_init,&(exact_init),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_objective",
                  "do not set the objective evaluation function",
                  "phelm.c",no_objective,&(no_objective),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-no_gradient",
                  "do not set the residual evaluation function",
                  "phelm.c",no_gradient,&(no_gradient),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p",
                  "exponent p with  1 <= p",
                  "phelm.c",user.p,&(user.p),NULL); CHKERRQ(ierr);
    if (user.p < 1.0) {
         SETERRQ(PETSC_COMM_WORLD,1,"p >= 1 required");
    }
    ierr = PetscOptionsEnum("-problem",
                  "problem type determines right side f(x,y)",
                  "phelm.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,
                  NULL); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-quadpts",
                  "number n of quadrature points in each direction (= 1,2,3 only)",
                  "phelm.c",user.quadpts,&(user.quadpts),NULL); CHKERRQ(ierr);
    if ((user.quadpts < 1) || (user.quadpts > 3)) {
        SETERRQ(PETSC_COMM_WORLD,3,"quadrature points n=1,2,3 only");
    }
    ierr = PetscOptionsBool("-view_f",
                  "view right-hand side to STDOUT",
                  "phelm.c",view_f,&(view_f),NULL);CHKERRQ(ierr);
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
    if (no_gradient) {
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
                ierr = GetVecFromFunction(&info,u_initial,&u_exact_cosines,&user); CHKERRQ(ierr);
            }
            user.f = &f_cosines;
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,4,"unknown problem type\n");
    }

    // optionally view right-hand-side on initial grid
    if (view_f) {
        Vec vf;
        ierr = VecDuplicate(u_initial,&vf); CHKERRQ(ierr);
        switch (problem) {
            case CONSTANT:
                ierr = VecSet(vf,1.0); CHKERRQ(ierr);
                break;
            case COSINES:
                ierr = GetVecFromFunction(&info,vf,&f_cosines,&user); CHKERRQ(ierr);
                break;
        }
        ierr = VecView(vf,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        VecDestroy(&vf);
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
            ierr = GetVecFromFunction(&info,u_exact,&u_exact_cosines,&user); CHKERRQ(ierr);
            break;
    }
    ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
        "done on %d x %d grid; p=%.3f; numerical error:  |u-u_exact|_inf = %.3e\n",
        info.mx,info.my,user.p,err); CHKERRQ(ierr);

    VecDestroy(&u_exact);  SNESDestroy(&snes);
    return PetscFinalize();
}

PetscErrorCode GetVecFromFunction(DMDALocalInfo *info, Vec w,
                   double (*fcn)(double, double, double, double), PHelmCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0 / (info->mx - 1), hy = 1.0 / (info->my - 1);
    double       x, y, **aw;
    int          i, j;
    ierr = DMDAVecGetArray(info->da,w,&aw); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            aw[j][i] = (*fcn)(x,y,user->p,user->eps);
        }
    }
    ierr = DMDAVecRestoreArray(info->da,w,&aw); CHKERRQ(ierr);
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
    return   v[0] * chi(0,xi,eta) + v[1] * chi(1,xi,eta)
           + v[2] * chi(2,xi,eta) + v[3] * chi(3,xi,eta);
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
    return cx * du.xi * dv.xi + cy * du.eta * dv.eta;
}

static double GradPow(double hx, double hy,
                      gradRef du, double P, double eps) {
    return PetscPowScalar(GradInnerProd(hx,hy,du,du) + eps*eps, P/2.0);
}
//ENDFEM

/* FLOPS:  (counting PetscPowScalar as 1)
     chi = 6
     eval = 4*6+7 = 31
     dchi = 8
     deval = 4*8+4 = 36
     GradInnerProd = 9
     GradPow = 9+4 = 13
     ObjIntegrandRef = deval + 2*eval + GradPow + 10 = 121
     FunIntegrandRef = chi + dchi + 2*eval + deval + GradPo + GradInnerProd + 9
                     = 143
*/

//STARTOBJECTIVE
static double ObjIntegrandRef(DMDALocalInfo *info,
                       const double ff[4], const double uu[4],
                       double xi, double eta, PHelmCtx *user) {
    const gradRef du = deval(uu,xi,eta);
    const double  hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1),
                  u = eval(uu,xi,eta);
    return GradPow(hx,hy,du,user->p,0.0) / user->p + 0.5 * u * u
           - eval(ff,xi,eta) * u;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, double **au,
                                  double *obj, PHelmCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  const Quad1D q = gausslegendre[user->quadpts-1];
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
                          * ObjIntegrandRef(info,ff,uu,
                                            q.xi[r],q.xi[s],user);
              }
          }
      }
  }
  lobj *= hx * hy / 4.0;  // from change of variables formula
  ierr = PetscObjectGetComm((PetscObject)(info->da),&com); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPI_DOUBLE,MPI_SUM,com); CHKERRQ(ierr);
  ierr = PetscLogFlops(129*info->xm*info->ym); CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  const double hx = 1.0 / (info->mx - 1),  hy = 1.0 / (info->my - 1);
  const Quad1D q = gausslegendre[user->quadpts-1];
  const int    li[4] = {0,-1,-1,0},  lj[4] = {0,0,-1,-1};
  double       x, y;
  int          i,j,l,r,s,PP,QQ;

  // clear residuals
  for (j = info->ys; j < info->ys + info->ym; j++)
      for (i = info->xs; i < info->xs + info->xm; i++)
          FF[j][i] = 0.0;

  // loop over all elements
  for (j = info->ys; j <= info->ys + info->ym; j++) {
      if ((j == 0) || (j > info->my-1))
          continue;
      y = j * hy;
      for (i = info->xs; i <= info->xs + info->xm; i++) {
          if ((i == 0) || (i > info->mx-1))
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
  ierr = PetscLogFlops((5+q.n*q.n*149)*(info->xm+1)*(info->ym+1)); CHKERRQ(ierr);
  return 0;
}
//ENDFUNCTION

