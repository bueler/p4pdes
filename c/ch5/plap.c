static char help[] = "Solve the p-laplacian equation in 2D using an objective function.\n\n";

// RUN AS
//   ./plap -snes_fd_function   (no residual = gradient function evaluation)
//   ./plap -snes_fd_color   (no Jacobian = Hessian function evaluation)

#include <petsc.h>

//STARTCONFIGURE
typedef struct {
  PetscReal p, dx, dy;
  PetscBool manufactured;
  Vec       f;
} PLapCtx;

PetscErrorCode Configure(PLapCtx *user) {
  PetscErrorCode ierr;
  user->p = 2.0;
  user->manufactured = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                   NULL,user->p,&(user->p),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-manufactured","use manufactured solution (p=2,4 only)",
                   NULL,user->manufactured,&(user->manufactured),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if ((user->manufactured) && (user->p != 2.0) && (user->p != 4.0)) {
      SETERRQ1(PETSC_COMM_WORLD,1,"no manufactured soln for p=%.3f",user->p);
  }
  return 0;
}

PetscErrorCode PrintResult(DMDALocalInfo *info, SNES snes, Vec u, Vec uexact,
                           PLapCtx *user) {
  PetscErrorCode ierr;
  PetscReal            unorm, errnorm;
  SNESConvergedReason  reason;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"on %d x %d grid:  ",info->mx,info->my); CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason); CHKERRQ(ierr);
  if (user->manufactured) {
      ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
      ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
      ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"|u-u_exact|_inf/|u|_inf = %g\n",
                  errnorm/unorm); CHKERRQ(ierr);
  } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%s\n",SNESConvergedReasons[reason]); CHKERRQ(ierr);
  }
  return 0;
}
//ENDCONFIGURE

//STARTEXACTF
PetscErrorCode ExactFLocal(DMDALocalInfo *info,
                           PetscReal **uex, PetscReal **f, PLapCtx *user) {
  PetscInt         i,j;
  PetscReal        x,y,s,c,ux,uy,py,dpy,gs,gsx,gsy,lap;
  const PetscReal  pi = PETSC_PI, pi2 = pi * pi, pi3 = pi2 * pi;
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y   = user->dy * j;
    py  = y * (1.0 - y);
    dpy = 1.0 - 2.0 * y;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = user->dx * i;
      s = sin(2.0*pi*x);
      if (user->manufactured) {
          uex[j][i] = s * py;  //  u(x,y) = sin(2 pi x) y (1 - y)
          lap = 2.0 * s * (2.0 * pi2 * py + 1.0);           // = u_xx + u_yy
          if (user->p == 2.0) {
            f[j][i] = - lap;
          } else if (user->p == 4.0) {
            c = cos(2.0*pi*x);
            ux  = 2.0 * pi * c * py;
            uy  = s * dpy;
            gs  = 4.0 * pi2 * c*c * py*py + s*s * dpy*dpy;  // = |grad u|^2
            gsx = - 16.0 * pi3 * c*s * py*py + 4.0 * pi * s*c * dpy*dpy;
            gsy = 4.0 * pi2 * c*c * 2.0 * py*dpy + s*s * 2.0 * dpy*(-2.0);
            f[j][i] = - gsx * ux - gsy * uy - gs * lap;
          } else {
            SETERRQ(PETSC_COMM_WORLD,1,"HOW DID I GET HERE?");
          }
      } else {
        uex[j][i] = NAN;
        f[j][i] = s * py;  //  f(x,y) = sin(2 pi x) y (1 - y)
      }
    }
  }
  return 0;
}
//ENDEXACTF

//STARTOBJECTIVE
static PetscReal xiell  = {-1.0, +1.0, +1.0, -1.0},
                 etaell = {-1.0, -1.0, +1.0, +1.0},
                 xiq    = {}, // FIXME: fix quad degree n=2
                 etaq   = {};

PetscReal chi(PetscInt l, PetscReal xi, PetscReal eta) {
  return 0.25 * (1.0 + ;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **u,
                                  PetscReal *obj, PLapCtx *user) {
  PetscErrorCode   ierr;
  PetscReal        lobj = 0.0;
  PetscInt         i,j;
  const PetscReal  pi = PETSC_PI, pi2 = pi * pi, pi3 = pi2 * pi;
  for (j=info->ys; j<info->ys+info->ym; j++) {
      if (j == info->my - 1) continue;
      for (i=info->xs; i<info->xs+info->xm; i++) {
          if (i == info->mx - 1) continue;
          
          lobj += ;
      }
  }
  lobj *= 0.25 * hx * hy;
  ierr = MPI_Allreduce(&lobj,obj,1,MPIU_REAL,MPIU_SUM,PETSC_COMM_WORLD); CHKERRQ(ierr);
  return 0;
}
//ENDOBJECTIVE

//STARTFUNCTION
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **f, PLapCtx *user) {
  SETERRQ(PETSC_COMM_WORLD,1,"NOT YET IMPLEMENTED");
  return 0;
}
//ENDFUNCTION

//STARTMAIN
int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM                   da;
  SNES                 snes;
  Vec                  u, uexact;
  PetscReal            **auexact, **af;
  PLapCtx              user;
  DMDALocalInfo        info;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = Configure(&user); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  user.dx = 1.0 / (PetscReal)(info.mx-1);
  user.dy = 1.0 / (PetscReal)(info.my-1);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.f));CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,uexact,&auexact); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,user.f,&af); CHKERRQ(ierr);
  ierr = ExactFLocal(&info,auexact,af,&user); CHKERRQ(ierr);
  VecSet(u,1.0);// FIXME: correctly initialize u
  ierr = DMDAVecRestoreArray(da,uexact,&auexact); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,user.f,&af); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = PrintResult(&info,snes,u,uexact,&user); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN

