static char help[] = "Solve the p-laplacian equation in 2D using Q^1 FEM\n"
"and an objective function.  Uses a manufactured solution.\n\n";

// RUN AS
//   ./plap -snes_fd
//   ./plap -snes_mf
//   ./plap -snes_fd_function -snes_fd
// because there is no Jacobian (= Hessian)

// EVIDENCE OF CONVERGENCE WITH OBJECTIVE-ONLY (note diverged linear solve and crappy errors):
// for LEV in 0 1 2; do ./plap -snes_fd_function -snes_fd -snes_monitor -snes_converged_reason -ksp_type preonly -pc_type cholesky -da_refine $LEV; done

// CHECK SNES CONVERGENCE WITH -snes_fd_color AND LINEAR CASE p=2:
// timer ./plap -snes_fd_color -ksp_rtol 1.0e-14 -ksp_converged_reason -snes_monitor -snes_converged_reason -ksp_type cg -pc_type icc -da_refine 6

// EVIDENCE OF CONVERGENCE WITH -snes_fd_color AND IN PARALLEL AND LINEAR CASE (p=2):
// for LEV in 0 1 2 3 4 5 6 7; do mpiexec -n 4 ./plap -snes_fd_color -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_rtol 1.0e-14 -ksp_converged_reason -snes_converged_reason -da_refine $LEV; done

// EVIDENCE OF CONVERGENCE WITH -snes_fd_color NOT PARALLEL AND NONLINEAR CASE (p=4):
// (note number of snes iterations grows)
// for LEV in 0 1 2 3 4 ; do ./plap -snes_fd_color -plap_p 4 -ksp_type preonly -pc_type cholesky -snes_converged_reason -da_refine $LEV; done

#include <petsc.h>

#define COMM PETSC_COMM_WORLD

//STARTDECLARE
typedef struct {
  DM        da;
  PetscReal p;
  Vec       f,g;
} PLapCtx;

PetscReal BoundaryG(PetscReal x, PetscReal y) {
    return 0.5 * (x+1.0)*(x+1.0) * (y+1.0)*(y+1.0);
}
//ENDDECLARE

//STARTBOUNDARY
PetscErrorCode SetGLocal(DMDALocalInfo *info, Vec g, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  PetscInt        i,j;
  PetscReal       x,y, **ag;

  ierr = DMDAVecGetArray(user->da,g,&ag); CHKERRQ(ierr);
  for (j = info->ys-1; j <= info->ys + info->ym; j++) {
      y = hy * (j + 1);
      for (i = info->xs-1; i <= info->xs + info->xm; i++) {
          x = hx * (i + 1);
          if ((j == -1) || (j == info->my)) {
              ag[j][i] = BoundaryG(x,y);    // bottom or top
          } else if (i == -1) {
              ag[j][i] = BoundaryG(0.0,y);  // left
          } else if (i == info->mx) {
              ag[j][i] = BoundaryG(1.0,y);  // right
          } else {
              ag[j][i] = NAN;
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
//ENDBOUNDARY

//STARTEXACT
PetscErrorCode ExactRHSLocal(DMDALocalInfo *info, Vec uex, Vec f, PLapCtx *user) {
    PetscErrorCode ierr;
    const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1),
                    p = user->p;
    const PetscInt  XE = info->xs + info->xm, YE = info->ys + info->ym;
    PetscInt        i,j;
    PetscReal       x,y, XX,YY, C,D2,gamma1,gamma2, **auex, **af;

// PetscPrintf(COMM,"xs=%d, xm=%d, mx=%d\n",info->xs,info->xm,info->mx);
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
            af[j][i] = - (p-2.0) * C * (gamma1*(x+1.0)*YY + gamma2*XX*(y+1.0)) - C * D2;
            if ((i >= info->xs) && (i < XE) && (j >= info->ys) && (j < YE)) {
                auex[j][i] = BoundaryG(x,y);
            }
        }
    }
    ierr = DMDAVecRestoreArray(user->da,uex,&auex); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da,f,&af); CHKERRQ(ierr);
    return 0;
}
//ENDEXACT

//STARTFEM
static PetscReal zq[2] = {-0.577350269189626,0.577350269189626},
                 wq[2] = {1.0,1.0};

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

// evaluate v(xi,eta) on ref. element using local node numbering
PetscReal eval(const PetscReal v[4], PetscReal xi, PetscReal eta) {
  PetscReal sum = 0.0;
  PetscInt  L;
  for (L=0; L<4; L++)
    sum += v[L] * chi(L,xi,eta);
  return sum;
}

// evaluate partial derivs of v(xi,eta) on ref. element
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
//ENDFEM

//STARTOBJECTIVE
PetscReal GradInnerProd(gradRef du, gradRef dv, DMDALocalInfo *info) {
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  PetscReal       z;
  z =  (4.0 / (hx * hx)) * du.xi  * dv.xi;
  z += (4.0 / (hy * hy)) * du.eta * dv.eta;
  return z;
}

PetscReal GradPow(gradRef du, PetscReal P, DMDALocalInfo *info) {
  return PetscPowScalar(GradInnerProd(du,du,info), P / 2.0);
}

PetscReal ObjIntegrand(const PetscReal f[4], const PetscReal u[4],
                       PetscReal xi, PetscReal eta, PetscReal P,
                       DMDALocalInfo *info) {
  const gradRef du = deval(u,xi,eta);
  return GradPow(du,P,info) / P - eval(f,xi,eta) * eval(u,xi,eta);
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  PetscReal      lobj = 0.0, **af, **ag, f[4], u[4];
  const PetscInt n = 2;  // FIXME: quad deg
  const PetscInt XE = info->xs + info->xm, YE = info->ys + info->ym;
  PetscInt       i,j,r,s;
  MPI_Comm       com;

  // sum over all elements
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,user->g,&ag); CHKERRQ(ierr);
  for (j = info->ys; j <= YE; j++) {
      for (i = info->xs; i <= XE; i++) {
          if ((i < XE) || (j < YE) || (i == info->mx) || (j == info->my)) {
//FIXME: must modify the u evals to use g
              // because of ghosts, these values are always valid even in the
              //     "right" and "top" cases that where i==info->mx or j==info->my
              f[0] = af[j][i];  f[1] = af[j][i-1];
                  f[2] = af[j-1][i-1];  f[3] = af[j-1][i];
              u[0] = au[j][i];  u[1] = au[j][i-1];
                  u[2] = au[j-1][i-1];  u[3] = au[j-1][i];
              for (r=0; r<n; r++) {
                  for (s=0; s<n; s++) {
                      lobj += wq[r] * wq[s] * ObjIntegrand(f,u,zq[r],zq[s],user->p,info);
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
PetscReal FunIntegrand(PetscInt L,
                       const PetscReal f[4], const PetscReal u[4],
                       PetscReal xi, PetscReal eta, PetscReal P,
                       DMDALocalInfo *info) {
  const gradRef du    = deval(u,xi,eta),
                dchiL = dchi(L,xi,eta);
  return GradPow(du,P - 2.0,info) * GradInnerProd(du,dchiL,info)
         - eval(f,xi,eta) * chi(L,xi,eta);
  return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (info->mx+1), hy = 1.0 / (info->my+1);
  PetscReal       **af, **ag, f[4], u[4], z;
  const PetscInt  n = 2;  // FIXME: quad deg
  PetscInt        i,j,c,d,r,s;
  const PetscInt  ell[2][2] = { {0,3}, {1,2} };

  // compute residual FF[j][i] for each node (x_i,y_j)
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,user->g,&ag); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          // sum over four elements which contribute to current node
          z = 0.0;
          for (c = 0; c < 2; c++) {
              for (d = 0; d < 2; d++) {
//FIXME: must modify the u evals to use g
                  f[0] = af[j+d][i+c];  f[1] = af[j+d][i+c-1];
                      f[2] = af[j+d-1][i+c-1];  f[3] = af[j+d-1][i+c];
                  u[0] = au[j+d][i+c];  u[1] = au[j+d][i+c-1];
                      u[2] = au[j+d-1][i+c-1];  u[3] = au[j+d-1][i+c];
                  for (r=0; r<n; r++) {
                      for (s=0; s<n; s++) {
                          z += wq[r] * wq[s] * FunIntegrand(ell[c][d],f,u,zq[r],zq[s],user->p,info);
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
  user.p = 2.0;
  ierr = PetscOptionsBegin(COMM,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                   NULL,user.p,&(user.p),NULL); CHKERRQ(ierr);
  if (user.p < 1.0) { SETERRQ(COMM,1,"p >= 1 required"); }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX,
               -3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &(user.da)); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"grid of %d x %d = %d interior nodes (%dx%d elements)\n",
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
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
      "numerical error:  |u-u_exact|_inf/|u_exact|_inf = %g\n",err/unorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  VecDestroy(&(user.f));  VecDestroy(&(user.g));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}
//ENDMAIN

