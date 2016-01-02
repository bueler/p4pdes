static char help[] = "Solve the p-laplacian equation in 2D using Q^1 FEM\n"
"and an objective function.\n\n";

// RUN AS
//   ./plap -snes_fd
//   ./plap -snes_mf
//   ./plap -snes_fd_function -snes_fd
// because there is no Jacobian (= Hessian)

// EVIDENCE OF CONVERGENCE WITH OBJECTIVE-ONLY (note diverged linear solve and crappy errors):
// for LEV in 0 1 2; do ./plap -snes_fd_function -snes_fd -plap_manufactured -snes_monitor -snes_converged_reason -ksp_type preonly -pc_type cholesky -da_refine $LEV; done

// CHECK SNES CONVERGENCE WITH -snes_fd_color AND LINEAR CASE p=2:
// timer ./plap -snes_fd_color -plap_manufactured -ksp_rtol 1.0e-14 -ksp_converged_reason -snes_monitor -snes_converged_reason -ksp_type cg -pc_type icc -da_refine 6

// EVIDENCE OF CONVERGENCE WITH -snes_fd_color AND IN PARALLEL AND LINEAR CASE (p=2):
// for LEV in 0 1 2 3 4 5 6 7; do mpiexec -n 4 ./plap -snes_fd_color -plap_manufactured -ksp_type cg -pc_type bjacobi -sub_pc_type icc -ksp_rtol 1.0e-14 -ksp_converged_reason -snes_converged_reason -da_refine $LEV; done

// EVIDENCE OF CONVERGENCE WITH -snes_fd_color NOT PARALLEL AND NONLINEAR CASE (p=4):
// (note number of snes iterations grows)
// for LEV in 0 1 2 3 4 ; do ./plap -snes_fd_color -plap_manufactured -plap_p 4 -ksp_type preonly -pc_type cholesky -snes_converged_reason -da_refine $LEV; done

#include <petsc.h>

#define COMM PETSC_COMM_WORLD

//STARTCONFIGURE
typedef struct {
  DM        da;
  PetscReal p;
  PetscBool manufactured;
  Vec       f;
} PLapCtx;

PetscErrorCode Configure(PLapCtx *user) {
  PetscErrorCode ierr;
  user->p = 2.0;
  user->manufactured = PETSC_FALSE;
  ierr = PetscOptionsBegin(COMM,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                   NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
  if (user->p < 1.0) {
      SETERRQ1(COMM,1,"p=%.3f invalid ... p >= 1 required",user->p);
  }
  ierr = PetscOptionsBool("-manufactured","use manufactured solution (p=2,4 only)",
                   NULL,user->manufactured,&(user->manufactured),NULL);CHKERRQ(ierr);
  if ((user->manufactured) && (user->p != 2.0) && (user->p != 4.0)) {
      SETERRQ1(COMM,2,"no manufactured soln for p=%.3f; use p=2,4",user->p);
  }
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  return 0;
}
//ENDCONFIGURE

//STARTBOUNDARY
PetscErrorCode SetBoundaryLocal(DMDALocalInfo *info, PetscReal **au) {
  const PetscInt XE = info->xs + info->xm, YE = info->ys + info->ym;
  PetscInt       i,j;
  for (j = info->ys-1; j <= YE; j++) {
      if ((j == -1) || (j == info->mx)) {
          for (i = info->xs-1; i <= XE; i++)
              au[j][i] = 0.0;    // top and bottom boundary values
      } else if (info->xs == 0)
          au[j][-1] = 0.0;       // left boundary values
      else if (XE == info->mx)
          au[j][info->mx] = 0.0; // right boundary values
  }
  return 0;
}

PetscErrorCode InitialValues(DMDALocalInfo *info, Vec u, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal  hx = 1.0 / (PetscReal)(info->mx+1),
                   hy = 1.0 / (PetscReal)(info->my+1);
  PetscInt         i,j;
  PetscReal        x,y, **au;

  ierr = DMDAVecGetArray(user->da,u,&au); CHKERRQ(ierr);
  // loop over interior grid points, including ghosts
  for (j = info->ys; j < info->ys + info->ym; j++) {
    y = hy * (j + 1);
    for (i = info->xs; i < info->xs + info->xm; i++) {
      x = hx * (i + 1);
      au[j][i] = x * (1.0 - x) * y * (1.0 - y);  // positive in interior
    }
  }
  ierr = DMDAVecRestoreArray(user->da,u,&au); CHKERRQ(ierr);
  return 0;
}
//ENDBOUNDARY

//STARTEXACT
PetscErrorCode ExactLocal(DMDALocalInfo *info, Vec uex, Vec f,
                          PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal  hx = 1.0 / (PetscReal)(info->mx+1),
                   hy = 1.0 / (PetscReal)(info->my+1);
  const PetscInt   XE = info->xs + info->xm, YE = info->ys + info->ym;
  PetscInt         i,j;
  PetscReal        x,y, x2,x4,y2,y4, px,py, ux,uy, uxx,uxy,uyy,lap,
                   gs,gsx,gsy, **auex, **af;

  ierr = DMDAVecGetArray(user->da,uex,&auex); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,f,&af); CHKERRQ(ierr);
  // loop over ALL grid points, including ghosts
  // note f is local (has ghosts) but uex is global (no ghosts)
  for (j = info->ys - 1; j <= YE; j++) {
    y   = hy * (j + 1);  y2  = y * y;  y4  = y2 * y2;
    py  = y4 - y2;                                 // polynomial in x
    for (i = info->xs - 1; i <= XE; i++) {
      x   = hx * (i + 1);  x2  = x * x;  x4  = x2 * x2;
      px  = x2 - x4;                               // polynomial in y
      uxx = 2.0 * (1.0 - 6.0 * x2) * py;
      uyy = 2.0 * (6.0 * y2 - 1.0) * px;
      lap = uxx + uyy;
      if (user->p == 2.0) {
        af[j][i] = - lap;
      } else if (user->p == 4.0) {
        ux  = 2.0 * x * (1.0 - 2.0 * x2) * py;
        uy  = 2.0 * y * (2.0 * y2 - 1.0) * px;
        gs  = ux * ux + uy * uy;                   // = |grad u|^2
        uxy = 4.0 * x * y * (1.0 - 2.0 * x2) * (2.0 * y2 - 1.0);
        gsx = 2.0 * (ux * uxx + uy * uxy);
        gsy = 2.0 * (ux * uxy + uy * uyy);
        af[j][i] = - gs * lap - ux * gsx - uy * gsy;
      } else {
        SETERRQ(COMM,1,"p!=2,4 ... HOW DID I GET HERE?");
      }
      if ((i >= info->xs) && (i < XE) && (j >= info->ys) && (j < YE)) {
        auex[j][i] = px * py;  //  u(x,y) = (x^2 - x^4) (y^4 - y^2)
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
  const PetscReal hx = 1.0 / (PetscReal)(info->mx+1),
                  hy = 1.0 / (PetscReal)(info->my+1);
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

static PetscReal zq[2] = {-0.577350269189626,0.577350269189626},
                 wq[2] = {1.0,1.0};

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0 / (PetscReal)(info->mx+1),
                  hy = 1.0 / (PetscReal)(info->my+1);
  PetscReal      lobj = 0.0, **af, f[4], u[4];
  const PetscInt n = 2;  // FIXME: quad deg
  const PetscInt XE = info->xs + info->xm, YE = info->ys + info->ym;
  PetscInt       i,j,r,s;
  MPI_Comm       com;

  ierr = SetBoundaryLocal(info,au); CHKERRQ(ierr);
  // sum over all elements
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  for (j = info->ys; j <= YE; j++) {
      for (i = info->xs; i <= XE; i++) {
          if ((i < XE) || (j < YE) || (i == info->mx) || (j == info->my)) {
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
  const PetscReal hx = 1.0 / (PetscReal)(info->mx+1),
                  hy = 1.0 / (PetscReal)(info->my+1);
  PetscReal      **af, f[4], u[4], z;
  const PetscInt n = 2;  // FIXME: quad deg
  PetscInt       i,j,c,d,r,s;
  const PetscInt ell[2][2] = { {0,3}, {1,2} };

  ierr = SetBoundaryLocal(info,au); CHKERRQ(ierr);
  // compute residual FF[j][i] for each node (x_i,y_j)
  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          // sum over four elements which contribute to current node
          z = 0.0;
          for (c = 0; c < 2; c++) {
              for (d = 0; d < 2; d++) {
                  f[0] = af[j+d][i+c];  f[1] = af[j+d][i+c-1];
                      f[2] = af[j+d-1][i+c-1];  f[3] = af[j+d-1][i+c];
                  u[0] = au[j+d][i+c];  u[1] = au[j+d][i+c-1];
                      u[2] = au[j+d-1][i+c-1];  u[3] = au[j+d-1][i+c];
                  //PetscPrintf(PETSC_COMM_WORLD,"c=%d,d=%d:  ell = %d\n",c,d,ell[c][d]);
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
  PetscReal      hx, hy;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = Configure(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX,
               -3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &(user.da)); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  hx = 1.0 / (PetscReal)(info.mx+1);
  hy = 1.0 / (PetscReal)(info.my+1);
  ierr = DMDASetUniformCoordinates(user.da,0.0+hx,1.0-hx,0.0+hy,1.0-hy,0.0,1.0); CHKERRQ(ierr);

  ierr = PetscPrintf(COMM,"grid of %d x %d = %d interior nodes (%dx%d elements)\n",
            info.mx,info.my,info.mx*info.my,info.mx+1,info.my+1); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(user.da,&(user.f));CHKERRQ(ierr);
  ierr = InitialValues(&info,u,&user); CHKERRQ(ierr);
  if (user.manufactured) {
    ierr = ExactLocal(&info,uexact,user.f,&user); CHKERRQ(ierr);
  } else {
    ierr = VecSet(user.f,1.0); CHKERRQ(ierr);
    ierr = VecSet(uexact,NAN); CHKERRQ(ierr);
  }

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(user.da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  if (user.manufactured) {
      PetscReal unorm, err;
      ierr = VecNorm(uexact,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&err); CHKERRQ(ierr);
      ierr = PetscPrintf(COMM,
          "numerical error:  |u-u_exact|_inf/|u_exact|_inf = %g\n",err/unorm); CHKERRQ(ierr);
  }

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}
//ENDMAIN

