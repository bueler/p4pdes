static char help[] = "Solve the p-laplacian equation in 2D using Q^1 FEM\n"
"and an objective function.\n\n";

// RUN AS
//   ./plap -snes_fd_function   (no residual = gradient function evaluation)
//   ./plap -snes_fd_color      (no Jacobian = Hessian function evaluation)

#include <petsc.h>

#define COMM PETSC_COMM_WORLD

//STARTCONFIGURE
typedef struct {
  DM        da;
  PetscReal p, hx, hy;
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

PetscErrorCode PrintResult(DMDALocalInfo *info, SNES snes, Vec u, Vec uexact,
                           PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal            unorm, errnorm;
  SNESConvergedReason  reason;

  ierr = PetscPrintf(COMM,"on %d x %d grid:  ",info->mx,info->my); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"%s\n",SNESConvergedReasons[reason]); CHKERRQ(ierr);
  if (user->manufactured) {
      ierr = VecNorm(uexact,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = PetscPrintf(COMM,"exact solution norm:   |u_exact|_inf   = %g\n",
                  unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(COMM,"numerical error norm:  |u-u_exact|_inf = %g\n",
                  errnorm); CHKERRQ(ierr);
  }
  return 0;
}
//ENDCONFIGURE

//STARTEXACT
PetscErrorCode ExactLocal(DMDALocalInfo *info, Vec uex, Vec f,
                          PLapCtx *user) {
  PetscErrorCode ierr;
  PetscInt         i,j;
  PetscReal        x,y, s,c,ux,uy,py,dpy,gs,gsx,gsy,lap,
                   **auex, **af;
  const PetscReal  pi = PETSC_PI, pi2 = pi * pi, pi3 = pi2 * pi;

  ierr = DMDAVecGetArray(user->da,uex,&auex); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,f,&af); CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y   = user->hy * j;
    py  = y * (1.0 - y);
    dpy = 1.0 - 2.0 * y;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = user->hx * i;
      s = sin(2.0*pi*x);
      if (user->manufactured) {
          auex[j][i] = s * py;  //  u(x,y) = sin(2 pi x) y (1 - y)
          lap = - 2.0 * s * (2.0 * pi2 * py + 1.0);         // = u_xx + u_yy
          if (user->p == 2.0) {
            af[j][i] = - lap;
          } else if (user->p == 4.0) {
            c = cos(2.0*pi*x);
            ux  = 2.0 * pi * c * py;
            uy  = s * dpy;
            gs  = 4.0 * pi2 * c*c * py*py + s*s * dpy*dpy;  // = |grad u|^2
            gsx = - 16.0 * pi3 * c*s * py*py + 4.0 * pi * s*c * dpy*dpy;
            gsy = 8.0 * pi2 * c*c * py*dpy - 4.0 * s*s * dpy;
            af[j][i] = - gsx * ux - gsy * uy - gs * lap;
          } else {
            SETERRQ(COMM,1,"p!=2,4 ... HOW DID I GET HERE?");
          }
      } else {
        auex[j][i] = NAN;
        af[j][i] = s * py;  //  f(x,y) = sin(2 pi x) y (1 - y)
      }
    }
  }
  ierr = DMDAVecRestoreArray(user->da,uex,&auex); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,f,&af); CHKERRQ(ierr);
  return 0;
}
//ENDEXACT

//STARTFEM
static PetscReal xiL[4]  = {-1.0,  1.0,  1.0, -1.0},
                 etaL[4] = {-1.0, -1.0,  1.0,  1.0};

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
PetscReal ObjIntegrand(PetscInt i, PetscInt j, PetscReal **af, PetscReal **au,
                       PetscReal xi, PetscReal eta, PLapCtx *user) {
  const PetscReal u[4] = {au[j][i],au[j][i+1],au[j+1][i+1],au[j+1][i]},
                  f[4] = {af[j][i],af[j][i+1],af[j+1][i+1],af[j+1][i]};
  const gradRef   du = deval(u,xi,eta);
  PetscReal       z;
  z =  (4.0 / (user->hx * user->hx)) * du.xi  * du.xi;
  z += (4.0 / (user->hy * user->hy)) * du.eta * du.eta;
  return PetscPowScalar(z,user->p/2.0) / user->p - eval(f,xi,eta) * eval(u,xi,eta);
}

static PetscReal zq[2] = {-0.577350269189626,0.577350269189626}, // FIXME: quad deg
                 wq[2] = {1.0,1.0};

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal      lobj = 0.0, **af;
  const PetscInt n = 2;  // FIXME: quad deg
  PetscInt       i,j,r,s;
  MPI_Comm       com;

  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
      if (j == info->my - 1) continue;
      for (i=info->xs; i<info->xs+info->xm; i++) {
          if (i == info->mx - 1) continue;
          for (r=0; r<n; r++) {
              for (s=0; s<n; s++) {
                  lobj += wq[r] * wq[s] * ObjIntegrand(i,j,af,au,zq[r],zq[s],user);
              }
          }
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  lobj *= 0.25 * user->hx * user->hy;

  ierr = PetscObjectGetComm((PetscObject)(info->da),&com);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,com); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
PetscReal FunIntegrand(PetscInt i, PetscInt j, PetscReal **af, PetscReal **au,
                       PetscReal xi, PetscReal eta) {
  SETERRQ(COMM,1,"NOT YET IMPLEMENTED");
  return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **f, PLapCtx *user) {
  SETERRQ(COMM,1,"NOT YET IMPLEMENTED");
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

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = Configure(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &(user.da)); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  user.hx = 1.0 / (PetscReal)(info.mx-1);
  user.hy = 1.0 / (PetscReal)(info.my-1);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.f));CHKERRQ(ierr);
  ierr = ExactLocal(&info,uexact,user.f,&user); CHKERRQ(ierr);
  VecSet(u,0.0);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(user.da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = PrintResult(&info,snes,u,uexact,&user); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}
//ENDMAIN

