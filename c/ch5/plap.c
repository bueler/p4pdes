static char help[] = "Solve the p-laplacian equation in 2D using an objective function.\n\n";

// RUN AS
//   ./plap -snes_fd_function   (no residual = gradient function evaluation)
//   ./plap -snes_fd_color   (no Jacobian = Hessian function evaluation)

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
  if (user->manufactured) {
      ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(COMM,"|u-u_exact|_inf/|u|_inf = %g\n",
                  errnorm/unorm); CHKERRQ(ierr);
  } else {
      ierr = PetscPrintf(COMM,"%s\n",SNESConvergedReasons[reason]); CHKERRQ(ierr);
  }
  return 0;
}
//ENDCONFIGURE

//STARTEXACTF
PetscErrorCode ExactFLocal(DMDALocalInfo *info, Vec uexact, Vec f, PLapCtx *user) {
  PetscErrorCode ierr;
  PetscInt         i,j;
  PetscReal        x,y, s,c,ux,uy,py,dpy,gs,gsx,gsy,lap,
                   **auex, **af;
  const PetscReal  pi = PETSC_PI, pi2 = pi * pi, pi3 = pi2 * pi;

  ierr = DMDAVecGetArray(user->da,uexact,&auex); CHKERRQ(ierr);
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
          lap = 2.0 * s * (2.0 * pi2 * py + 1.0);           // = u_xx + u_yy
          if (user->p == 2.0) {
            af[j][i] = - lap;
          } else if (user->p == 4.0) {
            c = cos(2.0*pi*x);
            ux  = 2.0 * pi * c * py;
            uy  = s * dpy;
            gs  = 4.0 * pi2 * c*c * py*py + s*s * dpy*dpy;  // = |grad u|^2
            gsx = - 16.0 * pi3 * c*s * py*py + 4.0 * pi * s*c * dpy*dpy;
            gsy = 4.0 * pi2 * c*c * 2.0 * py*dpy + s*s * 2.0 * dpy*(-2.0);
            af[j][i] = - gsx * ux - gsy * uy - gs * lap;
          } else {
            SETERRQ(COMM,1,"p!=2 and p!=4 ... HOW DID I GET HERE?");
          }
      } else {
        auex[j][i] = NAN;
        af[j][i] = s * py;  //  f(x,y) = sin(2 pi x) y (1 - y)
      }
    }
  }
  ierr = DMDAVecRestoreArray(user->da,uexact,&auex); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,f,&af); CHKERRQ(ierr);
  return 0;
}
//ENDEXACTF

//STARTOBJECTIVE
static PetscInt  xi_shift[4]  = {0,  1,  1,  0},
                 eta_shift[4] = {0,  0,  1,  1};
static PetscReal zq[2]     = {-0.577350269189626,0.577350269189626}, // FIXME: fix quad degree n=2
                 wq[2]     = {1.0,1.0};

PetscReal chi(PetscInt l, PetscReal xi, PetscReal eta) {
  const PetscInt  xi_l  = 2 * xi_shift[l]  - 1,   // in {-1,1}
                  eta_l = 2 * eta_shift[l] - 1;   // in {-1,1}
  return 0.25 * (1.0 + xi_l * xi) * (1.0 + eta_l * eta);
}

// FIXME: add this:
//PetscReal dchi(PetscInt l, PetscReal xi, PetscReal eta) {

// evaluate the function  v(x,y)  on  \square_{i,j}  using local coords xi,eta
PetscReal refeval(PetscInt i, PetscInt j, PetscReal **v, PetscReal xi, PetscReal eta) {
  PetscReal sum = 0.0;
  PetscInt  l;
  for (l=0; l<4; l++) {
    sum += v[j + xi_shift[l]][i + eta_shift[l]] * chi(l,xi,eta);
  }
  return sum;
}

PetscReal integrand(PetscInt i, PetscInt j, PetscReal **af, PetscReal **au,
                    PetscReal xi, PetscReal eta) {
  return refeval(i,j,af,xi,eta) * refeval(i,j,au,xi,eta); // MAJOR FIXME
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **au,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal      lobj = 0.0, **af;
  PetscInt       i,j,r,s;
  MPI_Comm       comm;

  ierr = DMDAVecGetArray(user->da,user->f,&af); CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
      if (j == info->my - 1) continue;
      for (i=info->xs; i<info->xs+info->xm; i++) {
          if (i == info->mx - 1) continue;
          for (r=0; r<2; r++) {
              for (s=0; s<2; s++) {
                  lobj += wq[r] * wq[s] * integrand(i,j,af,au,zq[r],zq[s]);
              }
          }
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->f,&af); CHKERRQ(ierr);
  lobj *= 0.25 * user->hx * user->hy;

  ierr = PetscObjectGetComm((PetscObject)(info->da),&comm);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,comm); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
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
  ierr = ExactFLocal(&info,uexact,user.f,&user); CHKERRQ(ierr);
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

