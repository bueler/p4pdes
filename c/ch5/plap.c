static char help[] = "Solve the p-laplacian equation in 2D using an objective function.\n\n";

// RUN AS   ./plap -snes_fd_function   IF FUNCTION NOT IMPLEMENTED

#include <petsc.h>

//STARTFUNCTIONS
typedef struct {
  PetscReal p, dx, dy;
  Vec       f;
} PLapCtx;

PetscErrorCode ExactFLocal(DMDALocalInfo *info, PetscReal **uex, PetscReal **f,
                           PLapCtx *user) {
  PetscInt         i,j;
  PetscReal        x,y,s,c,paray,gradsqr;
  const PetscReal  pi2 = PETSC_PI * PETSC_PI;
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = user->dy * j;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = user->dx * i;
      s = sin(2.0*PETSC_PI*x);
      c = cos(2.0*PETSC_PI*x);
      paray = y * (1.0 - y);
      uex[j][i] = s * paray;
      gradsqr = 4.0 * pi2 * c*c * paray*paray + s*s * (1.0-2.0*y)*(1.0-2.0*y);
      f[j][i] = 0.0 * gradsqr;  // FIXME: perhaps only p=2,4?
    }
  }
  return 0;
}

PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscReal **u,
                                  PLapCtx *user) {
  PetscInt i,j;
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      u[j][i] = 0.0; // FIXME
    }
  }
  return 0;
}
//ENDFUNCTIONS

//STARTMAIN
int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  Vec            u, uexact;
  PetscReal      **auexact, **af, unorm, errnorm;
  PLapCtx        user;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,(char*)0,help);

  user.p = 2.0;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"plap_","p-laplacian solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-p","exponent p with  1 <= p < infty",
                   NULL,user.p,&(user.p),NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

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
  // FIXME: initialize u
  ierr = DMDAVecRestoreArray(da,uexact,&auexact); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,user.f,&af); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetObjectiveLocal(da,
             (DMDASNESObjective)FormObjectiveLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

  ierr = VecNorm(u,NORM_INFINITY,&unorm); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
              "on %d x %d grid:  |u-u_exact|_inf/|u|_inf = %g\n",
              info.mx,info.mx,errnorm/unorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}
//ENDMAIN
