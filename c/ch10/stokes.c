
static char help[] =
"Solves a structured-grid Stokes problem with DMDA and SNES using finite elements: Q^2-P^-1.\n"
"FIXME: START OVER\n"
"\n\n";

#include <petsc.h>


typedef struct {
  double u;
  double v;
  double p;
} Field;

typedef struct {
  DM        da;
  Vec       xexact;
  double    L,     // length of domain in x direction
            H,     // length of domain in y direction
            g1,    // signed component of gravity in x-direction
            g2,    // signed component of gravity in y-direction
            mu;    // viscosity
} AppCtx;

extern PetscErrorCode FormExactSolution(AppCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,AppCtx*);
//extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*,Field**,Mat,Mat,AppCtx*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         user;                         /* user-defined work context */
  SNES           snes;                         /* nonlinear solver */
  Vec            x;                            /* solution vector */

  PetscInitialize(&argc,&argv,(char*)0,help);

  const double rg = 1000.0 * 9.81, // = rho g; scale for body force;
                                      //     kg / (m^2 s^2);  weight of water
                  theta = PETSC_PI / 9.0; // 20 degrees
  user.L     = 1.0;
  user.H     = 1.0;
  user.g1    = rg * sin(theta);
  user.g2    = - rg * cos(theta);
  user.nu    = 0.25;  // Pa s;  typical dynamic viscosity of motor oil
  user.ppeps = 1.0;

// FIXME START OVER

  PetscBool doerror = PETSC_FALSE, exactinit = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "stokes_", "options for stokes", ""); CHKERRQ(ierr);
  FIXME
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      FIXME); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.H, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&x); CHKERRQ(ierr);

  ierr = DMDASetFieldName(user.da,0,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,1,"v"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,2,"p"); CHKERRQ(ierr);

  FIXME

  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = DMDestroy(&user.da); CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


PetscErrorCode FormExactSolution(AppCtx* user) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  int       i,j;
  double      hy, y;
  Field          **ax;

  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  hy = user->H / (double)(info.my - 1);
  ierr = DMDAVecGetArray(user->da,user->xexact,&ax); CHKERRQ(ierr);
  for (j = info.ys; j < info.ys+info.ym; j++) {
    y = hy * (double)j;
    for (i = info.xs; i < info.xs+info.xm; i++) {
      ax[j][i].u = (user->g1 / user->nu) * y * (user->H - y/2.0);
      ax[j][i].v = 0.0;
      ax[j][i].p = - user->g2 * (user->H - y);
    }
  }
  ierr = DMDAVecRestoreArray(user->da,user->xexact,&ax); CHKERRQ(ierr);
  return 0;
}

