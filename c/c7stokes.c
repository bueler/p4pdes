
static char help[] =
"Solves a structured-grid Stokes problem with DMDA and KSP using finite differences.\n"
"Domain is rectangle with stress-free top, zero-velocity bottom, and\n"
"periodic b.c.s in x.  Exact solution is known.\n"
"System matrix is made symmetric by extracting divergence\n"
"approximation for incompressibility equation from gradient approximation in\n"
"stress-balance equations.  Pressure-poisson equation allows adding Laplacian\n"
"of pressure into incompressibility equations.\n"
"\n\n";

#include <petscdmda.h>
#include <petscsnes.h>


typedef struct {
  PetscReal u;
  PetscReal v;
  PetscReal p;
} Field;

typedef struct {
  DM        da;
  PetscInt  dof;   // number of degrees of freedom at each node
  PetscReal L,     // length of domain in x direction
            H,     // length of domain in y direction
            g1,    // signed component of gravity in x-direction
            g2,    // signed component of gravity in y-direction
            nu,    // viscosity
            ppeps; // amount of Laplacian of pressure to add to incompressibility
} AppCtx;

extern PetscErrorCode FormInitialGuess(Vec,AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,AppCtx*);
//extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,Field**,Mat,Mat,AppCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;                         /* user-defined work context */
  SNES           snes;                         /* nonlinear solver */
  Vec            x;                            /* solution vector */

  PetscInitialize(&argc,&argv,(char*)0,help);

  user.dof   = 3;
  user.L     = 10.0;
  user.H     = 1.0;
  user.g1    = 1.0;
  user.g2    = -9.0;
  user.nu    = 1.0;
  user.ppeps = 1.0;

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                      -4,-4,PETSC_DECIDE,PETSC_DECIDE,
                      user.dof,1,NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.H, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&x); CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
                                  (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = FormInitialGuess(da,x,&user); CHKERRQ(ierr);
  //ierr = VecSet(x,0.0); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);

  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = DMDestroy(&user.da); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


// FIXME: this version only sets x to the exact solution
PetscErrorCode FormInitialGuess(Vec x, AppCtx* user) {
{
  PetscErrorCode ierr;
  DMDALocalInfo *info;
  PetscReal      **ax, hy, y;

  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  hy = user->H / (double)(info.my - 1);
  ierr = DMGetCoordinates(user->da,&xycoords); CHKERRQ(ierr);
  ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
  for (j = info.ys; j < info.ys+info.ym-1; j++) {
    y = hy * (double) j;
    for (i = info.xs; i < info.xs+info.xm-1; i++) {
      ax[j][i].u = (user->g1 / user->nu) * y * (user->H - y/2.0);
      ax[j][i].v = 0.0;
      ax[j][i].p = - user->g2 * (user->H - y);
    }
  }
  ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field **x, Field **f, AppCtx *user)
{
  Field          uLocal[3];
  PetscInt       i,j,k,l;
  PetscReal      hx, hy, uUP, vUP,
                 H = user->H, g1 = user->g1, g2 = user->g2,
                 nu = user->nu, eps = user->ppeps;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;

  hx = user->L / (PetscReal)(info->mx);    // periodic direction
  hy = user->H / (PetscReal)(info->my-1);  // non-periodic

  // zero-out locally-owned f;  from snes/examples/tutorials/ex7.c
  ierr = PetscMemzero((void*) &(f[info->xs][info->ys]),
                      info->xm*info->ym*sizeof(Field)); CHKERRQ(ierr);

  for (j = info->ys; j < info->ys+info->ym-1; j++) {
    for (i = info->xs; i < info->xs+info->xm-1; i++) {
      if (j == 0) {
        // Dirichlet conditions at bottom
        f[j][i].u = x[j][i].u;
        f[j][i].v = x[j][i].v;
        f[j][i].p = x[j][i].p + g2 * H;
      } else if (j == info.my-1) {
        // stress-free, and pressure zero, conditions at top
        uUP = FIXME;
        f[j][i].u = (uUP - x[j-1][i].u) / (2.0*hy);
        vUP = FIXME;
        f[j][i].v = (vUP - x[j-1][i].v) / hy;
        f[j][i].p = x[j][i].p;
      } else {
        // FD eqn I:
        f[j][i].u = - nu * (x[j][i+1].u - 2.0 * x[j][i].u + x[j][i-1].u) / (hx*hx)
                    - nu * (x[j+1][i].u - 2.0 * x[j][i].u + x[j+1][i].u) / (hy*hy)
                    + (x[j][i+1].p - x[j][i-1].p) / (2.0*hx)
                    - g1;
        // FD eqn II:
        f[j][i].v = - nu * (x[j][i+1].v - 2.0 * x[j][i].v + x[j][i-1].v) / (hx*hx)
                    - nu * (x[j+1][i].v - 2.0 * x[j][i].v + x[j+1][i].v) / (hy*hy)
                    + (x[j+1][i].p - x[j-1][i].p) / (2.0*hy)
                    - g2;
        // FD eqn III:
        f[j][i].p =   (x[j][i+1].u - x[j][i-1].u) / (2.0*hx)
                    + (x[j+1][i].v - x[j-1][i].v) / (2.0*hy)
                    - eps * (
                        (x[j][i+1].p - 2.0 * x[j][i].p + x[j][i-1].p) / (hx*hx)
                      + (x[j+1][i].p - 2.0 * x[j][i].p + x[j+1][i].p) / (hy*hy) );
    }
  }

  //ierr = PetscLogFlops(68.0*(info->ym-1)*(info->xm-1)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
