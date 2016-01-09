static char help[] = "Solve the p-Laplacian equation in 2D using Q^1 FEM.\n"
"Implements an objective function and a residual (gradient) function, but\n"
"no Jacobian.  Defaults to p=4 and quadrature degree n=2.  Run as one of:\n"
"   ./plap -snes_fd_color\n"
"   ./plap -snes_fd\n"
"   ./plap -snes_mf\n"
"   ./plap -snes_fd_function -snes_fd\n"
"Uses a manufactured solution.\n\n";

// EVIDENCE OF CONVERGENCE WITH OBJECTIVE-ONLY AND DIRECT SOLVE
// note diverged linear solve from bad gradient
// redo with -snes_fd_color to see total success
// this also tests symmetry and positive-definiteness because use Cholesky
// for LEV in 0 1 2; do ./plap -snes_fd_function -snes_fd -snes_monitor -snes_converged_reason -snes_linesearch_type basic -ksp_type preonly -pc_type cholesky -da_refine $LEV; done

// CHECK FINE-GRID SNES CONVERGENCE WITH -snes_fd_color:
// timer ./plap -snes_fd_color -ksp_converged_reason -snes_monitor -snes_converged_reason -ksp_type cg -pc_type icc -da_refine 6

// EVIDENCE OF CONVERGENCE WITH -snes_fd_color AND IN PARALLEL;
// bt line search fails in parallel?
// note nonlinear iterations increase with grid
// for LEV in 0 1 2 3 4 5 6 7; do mpiexec -n 4 ./plap -snes_fd_color -snes_linesearch_type basic -ksp_type cg -pc_type bjacobi -sub_pc_type icc -snes_converged_reason -da_refine $LEV; done

// SHOWS n=1 GIVES ROUGH DOUBLE ERROR OVER n=2; n=3 NO BENEFIT:
// for NN in 1 2 3; do for LEV in 0 2 4 6; do ./plap -snes_fd_color -snes_linesearch_type basic -ksp_type cg -pc_type icc -snes_converged_reason -da_refine $LEV -plap_quaddegree $NN; done; done


#include <petsc.h>

#define COMM PETSC_COMM_WORLD

//STARTCTX
typedef struct {
    DM        da;
    PetscReal p;
    PetscInt  quaddegree;
    Vec       f,g;
} PLapCtx;

PetscErrorCode ConfigureCtx(PLapCtx *user) {
    PetscErrorCode ierr;
    user->p = 4.0;
    user->quaddegree = 2;
    ierr = PetscOptionsBegin(COMM,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                      NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
    if (user->p < 1.0) { SETERRQ(COMM,1,"p >= 1 required"); }
    ierr = PetscOptionsInt("-quaddegree","quadrature degree n (= 1,2,3 only)",
                     NULL,user->quaddegree,&(user->quaddegree),NULL); CHKERRQ(ierr);
    if ((user->quaddegree < 1) || (user->quaddegree > 3)) {
        SETERRQ(COMM,2,"quadrature degree n=1,2,3 only"); }
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}
//ENDCTX

//STARTBDRYINIT
PetscReal BoundaryG(PetscReal x, PetscReal y) {
    return 0.5 * (x+1.0)*(x+1.0) * (y+1.0)*(y+1.0);
}

PetscErrorCode SetGLocal(DMDALocalInfo *info, Vec g, PLapCtx *user) {
    PetscErrorCode ierr;
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    PetscInt        i,j;
    PetscReal       x,y, **ag;

    ierr = DMDAVecGetArray(user->da,g,&ag); CHKERRQ(ierr);
    for (j = info->ys - 1; j <= info->ys + info->ym; j++) {
        for (i = info->xs - 1; i <= info->xs + info->xm; i++) {
            if ((i >= 0) && (i < info->mx) && (j >= 0) && (j < info->my)) {
                ag[j][i] = NAN;               // invalidate interior
            } else {
                x = hx * (i + 1);  y = hy * (j + 1);
                ag[j][i] = BoundaryG(x,y);    // along boundary
            }
        }
    }
    ierr = DMDAVecRestoreArray(user->da,g,&ag); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode InitialIterate(DMDALocalInfo *info, Vec u, PLapCtx *user) {
    PetscErrorCode ierr;
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    PetscInt        i,j;
    PetscReal       x,y, **au;

    ierr = DMDAVecGetArray(user->da,u,&au); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = hy * (j + 1);
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = hx * (i + 1);
            au[j][i] = (1.0 - x) * BoundaryG(0.0,y) + x * BoundaryG(1.0,y);
        }
    }
    ierr = DMDAVecRestoreArray(user->da,u,&au); CHKERRQ(ierr);
    return 0;
}
//ENDBDRYINIT

//STARTEXACT
PetscErrorCode ExactRHSLocal(DMDALocalInfo *info, Vec uex, Vec f,
                             PLapCtx *user) {
    PetscErrorCode ierr;
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1),
                    p = user->p;
    const PetscInt  XE = info->xs + info->xm, YE = info->ys + info->ym;
    PetscInt        i,j;
    PetscReal       x,y, XX,YY, C,D2,gamma1,gamma2, **auex, **af;

    ierr = DMDAVecGetArray(user->da,uex,&auex); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da,f,&af); CHKERRQ(ierr);
    // loop over ALL grid points; f has ghosts but uex does not
    for (j = info->ys - 1; j <= YE; j++) {
        y = hy * (j + 1);  YY = (y+1.0)*(y+1.0);
        for (i = info->xs - 1; i <= XE; i++) {
            x = hx * (i + 1);  XX = (x+1.0)*(x+1.0);
            D2 = XX + YY;
            C = PetscPowScalar(XX * YY * D2, (p - 2.0) / 2.0);
            gamma1 = 1.0/(x+1.0) + (x+1.0)/D2;
            gamma2 = 1.0/(y+1.0) + (y+1.0)/D2;
            af[j][i] = - (p-2.0) * C * (gamma1*(x+1.0)*YY + gamma2*XX*(y+1.0))
                       - C * D2;
            if ((i >= info->xs) && (i < XE) && (j >= info->ys) && (j < YE)) {
                auex[j][i] = BoundaryG(x,y);   // use as exact solution
            }
        }
    }
    ierr = DMDAVecRestoreArray(user->da,uex,&auex); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da,f,&af); CHKERRQ(ierr);
    return 0;
}
//ENDEXACT

//STARTFEM
static PetscReal xiL[4]  = { 1.0, -1.0, -1.0,  1.0},
                 etaL[4] = { 1.0,  1.0, -1.0, -1.0};

PetscReal chi(PetscInt L, PetscReal xi, PetscReal eta) {
    return 0.25 * (1.0 + xiL[L] * xi) * (1.0 + etaL[L] * eta);
}

typedef struct {
    PetscReal xi, eta;
} gradRef;

gradRef dchi(PetscInt L, PetscReal xi, PetscReal eta) {
    gradRef result;
    result.xi  = 0.25 * xiL[L]  * (1.0 + etaL[L] * eta);
    result.eta = 0.25 * etaL[L] * (1.0 + xiL[L]  * xi);
    return result;
}

// evaluate v(xi,eta) on reference element using local node numbering
PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    PetscReal sum = 0.0;
    PetscInt  L;
    for (L=0; L<4; L++)
        sum += v[L] * chi(L,xi,eta);
    return sum;
}

// evaluate partial derivs of v(xi,eta) on reference element
gradRef deval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
    gradRef   sum = {0.0,0.0}, tmp;
    PetscInt  L;
    for (L=0; L<4; L++) {
        tmp = dchi(L,xi,eta);
        sum.xi  += v[L] * tmp.xi;
        sum.eta += v[L] * tmp.eta;
    }
    return sum;
}

static PetscReal zq[3][3] = { {0.0,NAN,NAN},
                              {-1.0/sqrt(3.0),1.0/sqrt(3.0),NAN},
                              {-sqrt(3.0/5.0),0.0,sqrt(3.0/5.0)} },
                 wq[3][3] = { {2.0,NAN,NAN},
                              {1.0,1.0,NAN},
                              {5.0/9.0,8.0/9.0,5.0/9.0} };
//ENDFEM

//STARTTOOLS
void GetUorG(DMDALocalInfo *info, PetscInt i, PetscInt j,
             PetscReal **au, PetscReal **ag, PetscReal *u) {
    u[0] = ((i == info->mx) || (j == info->my))
             ? ag[j][i]     : au[j][i];
    u[1] = ((i == 0)  || (j == info->my))
             ? ag[j][i-1]   : au[j][i-1];
    u[2] = ((i == 0)  || (j == 0))
             ? ag[j-1][i-1] : au[j-1][i-1];
    u[3] = ((i == info->mx) || (j == 0))
             ? ag[j-1][i]   : au[j-1][i];
}

PetscReal GradInnerProd(DMDALocalInfo *info, gradRef du, gradRef dv) {
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
    PetscReal       z;
    z =  (4.0 / (hx * hx)) * du.xi  * dv.xi;
    z += (4.0 / (hy * hy)) * du.eta * dv.eta;
    return z;
}

PetscReal GradPow(DMDALocalInfo *info, gradRef du, PetscReal P) {
    return PetscPowScalar(GradInnerProd(info,du,du), P / 2.0);
}
//ENDTOOLS

//STARTOBJECTIVE
PetscReal ObjIntegrand(DMDALocalInfo *info,
                       const PetscReal f[4], const PetscReal u[4],
                       PetscReal xi, PetscReal eta, PetscReal P) {
    const gradRef du = deval(u,xi,eta);
    return GradPow(info,du,P) / P - eval(f,xi,eta) * eval(u,xi,eta);
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  const PetscInt  n = user->quaddegree,
                  XE = info->xs + info->xm, YE = info->ys + info->ym;
  PetscReal       lobj = 0.0, **af, **ag, f[4], u[4];
  PetscInt        i,j,r,s;
  MPI_Comm        com;

  // sum over all elements
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,user->g,&ag); CHKERRQ(ierr);
  for (j = info->ys; j <= YE; j++) {
      for (i = info->xs; i <= XE; i++) {
          if ((i < XE) || (j < YE) || (i == info->mx) || (j == info->my)) {
              // because of ghosts, these values are always valid even in the
              //     "right" and "top" cases where i==info->mx or j==info->my
              f[0] = af[j][i];  f[1] = af[j][i-1];
                  f[2] = af[j-1][i-1];  f[3] = af[j-1][i];
              GetUorG(info,i,j,au,ag,u);
              for (r=0; r<n; r++) {
                  for (s=0; s<n; s++) {
                      lobj += wq[n-1][r] * wq[n-1][s]
                              * ObjIntegrand(info,f,u,
                                             zq[n-1][r],zq[n-1][s],user->p);
                  }
              }
          }
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,user->g,&ag); CHKERRQ(ierr);
  lobj *= 0.25 * hx * hy;

  ierr = PetscObjectGetComm((PetscObject)(info->da),&com);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,com); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
PetscReal FunIntegrand(DMDALocalInfo *info, PetscInt L,
                       const PetscReal f[4], const PetscReal u[4],
                       PetscReal xi, PetscReal eta, PetscReal P) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return GradPow(info,du,P - 2.0) * GradInnerProd(info,du,dchiL)
         - eval(f,xi,eta) * chi(L,xi,eta);
  return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  const PetscInt  n = user->quaddegree,
                  ell[2][2] = { {0,3}, {1,2} };
  PetscReal       **af, **ag, f[4], u[4], z;
  PetscInt        i,j,c,d,r,s;

  // compute residual FF[j][i] for each node (x_i,y_j)
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,user->g,&ag); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          // sum over four elements which contribute to current node
          z = 0.0;
          for (c = 0; c < 2; c++) {
              for (d = 0; d < 2; d++) {
                  f[0] = af[j+d][i+c];  f[1] = af[j+d][i+c-1];
                      f[2] = af[j+d-1][i+c-1];  f[3] = af[j+d-1][i+c];
                  GetUorG(info,i+c,j+d,au,ag,u);
                  for (r=0; r<n; r++) {
                      for (s=0; s<n; s++) {
                          z += wq[n-1][r] * wq[n-1][s]
                               * FunIntegrand(info,ell[c][d],f,u,
                                              zq[n-1][r],zq[n-1][s],user->p);
                      }
                  }
              }
          }
          FF[j][i] = 0.25 * hx * hy * z;
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,user->g,&ag); CHKERRQ(ierr);
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
  PetscReal      unorm, err;

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = ConfigureCtx(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX,
               -3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &(user.da)); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "grid of %d x %d = %d interior nodes (%dx%d elements)\n",
            info.mx,info.my,info.mx*info.my,info.mx+1,info.my+1); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.da,&(user.f));CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.da,&(user.g));CHKERRQ(ierr);

  ierr = SetGLocal(&info,user.g,&user); CHKERRQ(ierr);
  ierr = InitialIterate(&info,u,&user); CHKERRQ(ierr);
  ierr = ExactRHSLocal(&info,uexact,user.f,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(user.da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecNorm(uexact,NORM_INFINITY,&unorm); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "numerical error:  |u-u_exact|_inf/|u_exact|_inf = %g\n",
           err/unorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  VecDestroy(&(user.f));  VecDestroy(&(user.g));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}
//ENDMAIN

