
static char help[] =
"Solves a structured-grid Stokes problem with DMDA and KSP using finite differences.\n"
"Domain is rectangle with stress-free top, zero-velocity bottom, and\n"
"periodic b.c.s in x.  Exact solution is known.\n"
"System matrix is made symmetric by extracting divergence\n"
"approximation for incompressibility equation from gradient approximation in\n"
"stress-balance equations.  Pressure-poisson equation allows adding Laplacian\n"
"of pressure into incompressibility equations.\n"
"\n\n";

// ./c7stokes -snes_fd -dm_view draw -draw_pause 2
// ./c7stokes -snes_fd -stokes_err -snes_monitor -snes_converged_reason
// ./c7stokes -snes_fd -stokes_err -snes_monitor -snes_converged_reason -da_grid_x 10 -da_grid_y 11  -snes_monitor_solution -draw_pause 2
// ./c7stokes -snes_fd -stokes_err -snes_monitor -snes_converged_reason -da_refine 3 -ksp_rtol 1.0e-14
// ./c7stokes -snes_fd -stokes_err -snes_monitor -snes_converged_reason -da_refine 3 -snes_rtol 1.0e-14

/* WORK ON SYMMETRY
./c7stokes -snes_fd -da_grid_x 2 -da_grid_y 6 -mat_view ::ascii_matlab >> foo.m
>> foo
>> A = Mat_0x1cd5470_0 
>> imagesc(A-A')
*/

// ./c7stokes -snes_fd -mat_is_symmetric 0.001    // FAILS FOR NOW

#include <petscdmda.h>
#include <petscsnes.h>


typedef struct {
  double u;
  double v;
  double p;
} Field;

typedef struct {
  DM        da;
  Vec       xexact;
  double L,     // length of domain in x direction
            H,     // length of domain in y direction
            g1,    // signed component of gravity in x-direction
            g2,    // signed component of gravity in y-direction
            nu,    // viscosity
            ppeps; // amount of Laplacian of pressure to add to incompressibility
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

// FIXME: TO DO:
//   3. use matlab output to work on making symmetric (e.g. on -da_grid_x 2 -da_grid_y 4
//      which gives A of size 24 x 24)
//   4. use matlab output to improve row scaling
//   5. add sliding exact solution from Will's thesis
//   6. add analytical jacobian
//   7. play with decreasing ppeps
//   8. is it symmetric even when hx != hy?

  PetscBool doerror = PETSC_FALSE, exactinit = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "stokes_", "options for c7stokes", ""); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-err",
           "evaluate and display numerical error (exact solution case)",
           "stokes.c",doerror,&doerror,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-exact_init",
           "use exact solution as initial value (instead of zero)",
           "stokes.c",exactinit,&exactinit,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-ppeps",
           "set epsilon which is the amount of Laplacian of pressure (i.e. pressure-poisson)",
           "stokes.c",user.ppeps,&user.ppeps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                      -4,-5,PETSC_DECIDE,PETSC_DECIDE,
                      3,1,NULL,NULL,&user.da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da, 0.0, user.H, 0.0, user.L, -1.0, -1.0); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&x); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,0,"u"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,1,"v"); CHKERRQ(ierr);
  ierr = DMDASetFieldName(user.da,2,"p"); CHKERRQ(ierr);

  // check that info.mx >= 4 so j==0,j==1,j==mx-2,j==mx-1 are different levels
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  if ((info.mx < 2) || (info.my < 4)) {
    SETERRQ2(PETSC_COMM_WORLD,1,
      "info.mx==%d and info.my=%d but implementation of boundary conditions\n"
      "  requires mx >= 2 and my >= 4\n",info.mx,info.my);
  }

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
                                  (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&user.xexact); CHKERRQ(ierr);
  ierr = FormExactSolution(&user); CHKERRQ(ierr);

  if (exactinit) {
    ierr = VecCopy(user.xexact,x); CHKERRQ(ierr);
  } else {
    ierr = VecSet(x,0.0); CHKERRQ(ierr);
  }

  ierr = SNESSolve(snes,NULL,x); CHKERRQ(ierr);

  if (doerror) {
    double  umax, vmax, pmax, uerr, verr, perr;
    ierr = VecStrideNorm(user.xexact,0,NORM_INFINITY,&umax); CHKERRQ(ierr);
    ierr = VecStrideNorm(user.xexact,1,NORM_INFINITY,&vmax); CHKERRQ(ierr);
    ierr = VecStrideNorm(user.xexact,2,NORM_INFINITY,&pmax); CHKERRQ(ierr);
    ierr = VecAXPY(x,-1.0,user.xexact); CHKERRQ(ierr);  // x := -xexact + x
    ierr = VecStrideNorm(x,0,NORM_INFINITY,&uerr); CHKERRQ(ierr);
    ierr = VecStrideNorm(x,1,NORM_INFINITY,&verr); CHKERRQ(ierr);
    ierr = VecStrideNorm(x,2,NORM_INFINITY,&perr); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                 "on %d x %d grid:\n"
                 "  |u - uexact|_inf = %e   (|uexact|_inf = %e)\n"
                 "  |v - vexact|_inf = %e   (|vexact|_inf = %e)\n"
                 "  |p - pexact|_inf = %e   (|pexact|_inf = %e)\n",
                 info.mx,info.my,uerr,umax,verr,vmax,perr,pmax); CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&user.xexact); CHKERRQ(ierr);
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


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field **x, Field **f, AppCtx *user) {
  int       i,j;
  double      hx, hy, ux, uxx, uyy, vxx, vy, vyy,
                 px, pxx, py, pyy, uUP, pUP,
                 H = user->H, g1 = user->g1, g2 = user->g2,
                 nu = user->nu, eps = user->ppeps;

  PetscFunctionBeginUser;

  hx = user->L / (double)(info->mx);    // periodic direction
  hy = user->H / (double)(info->my-1);  // non-periodic

  for (j = info->ys; j < info->ys+info->ym; j++) {
    for (i = info->xs; i < info->xs+info->xm; i++) {
      if (j == 0) {
        // Dirichlet conditions at bottom
        f[j][i].u = x[j][i].u;
        f[j][i].v = x[j][i].v;
        f[j][i].p = x[j][i].p + g2 * H;
      } else {
        // in all other cases we use some centered x-derivatives
        ux  = (x[j][i+1].u - x[j][i-1].u) / (2.0*hx);
        uxx = (x[j][i+1].u - 2.0 * x[j][i].u + x[j][i-1].u) / (hx*hx);
        vxx = (x[j][i+1].v - 2.0 * x[j][i].v + x[j][i-1].v) / (hx*hx);
        if (j == info->my-1) {
          // stress-free, and pressure zero, conditions at top
          uUP = 2.0 * x[j][i].u - x[j-1][i].u - (hy*hy) * ( (g1/nu) + uxx );
          f[j][i].u =   (uUP - x[j-1][i].u) / (2.0*hy)
                      + (x[j][i+1].v - x[j][i-1].v) / (2.0*hx);
          f[j][i].u *= nu / hy;  // enforce symmetry by matching coeff of u[j+1][i] below
          pUP = ((hy*hy) / eps) * ux - x[j-1][i].p;
          f[j][i].v = - nu * vxx
                      - nu * (2.0 * x[j-1][i].v - 2.0 * x[j][i].v) / (hy*hy)
                      + (pUP - x[j-1][i].p) / (2.0*hy) - g2;
          f[j][i].p = x[j][i].p;
        } else {
          // at generic points
          px  = (x[j][i+1].p - x[j][i-1].p) / (2.0*hx);
          pxx = (x[j][i+1].p - 2.0 * x[j][i].p + x[j][i-1].p) / (hx*hx);
          // use bottom p=-g2 H or top b.c. p=0 for symmetry
          if (j == 1) {
            py  = (x[j+1][i].p + g2 * H) / (2.0*hy);
            pyy = (x[j+1][i].p - 2.0 * x[j][i].p - g2 * H) / (hy*hy);
          } else if (j == info->my-2) {
            py  = (0.0 - x[j-1][i].p) / (2.0*hy);
            pyy = (0.0 - 2.0 * x[j][i].p + x[j-1][i].p) / (hy*hy);
          } else {
            py  = (x[j+1][i].p - x[j-1][i].p) / (2.0*hy);
            pyy = (x[j+1][i].p - 2.0 * x[j][i].p + x[j-1][i].p) / (hy*hy);
          }
          // use bottom u=0 for symmetry
          if (j == 1)
            uyy = (x[j+1][i].u - 2.0 * x[j][i].u + 0.0) / (hy*hy);
          else
            uyy = (x[j+1][i].u - 2.0 * x[j][i].u + x[j-1][i].u) / (hy*hy);
          // use bottom v=0 for symmetry
          if (j == 1) {
            vy  = (x[j+1][i].v - 0.0) / (2.0*hy);
            vyy = (x[j+1][i].v - 2.0 * x[j][i].v + 0.0) / (hy*hy);
          } else {
            vy  = (x[j+1][i].v - x[j-1][i].v) / (2.0*hy);
            vyy = (x[j+1][i].v - 2.0 * x[j][i].v + x[j-1][i].v) / (hy*hy);
          }
          // three field equations:
          f[j][i].u = - nu * (uxx + uyy) + px - g1;
          f[j][i].v = - nu * (vxx + vyy) + py - g2;
          f[j][i].p = - ux - vy - eps * (pxx + pyy);
        }
      }
    }
  }

  //ierr = PetscLogFlops(68.0*(info->ym-1)*(info->xm-1)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
