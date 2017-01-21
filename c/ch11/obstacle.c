static const char help[] = "Solves obstacle problem in 2D as a variational\n\
inequality.  An elliptic problem with solution  u  constrained to be above a\n\
given function  psi.  Exact solution is known.  Because of the constraint,\n\
the problem is nonlinear.\n";

/*
Parallel runs, spatial refinement only, robust PC:
  for M in 0 1 2 3 4 5 6; do
    mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu
  done
*/

#include <petsc.h>
#include "../ch6/jacobians.c"

typedef struct {
  DM  da;
  Vec psi,  // obstacle
      g;    // Dirichlet boundary conditions
} ObsCtx;

void GridSpaces(DMDALocalInfo *info, double *dx, double *dy) {
  *dx = 4.0 / (PetscReal)(info->mx-1);
  *dy = 4.0 / (PetscReal)(info->my-1);
}

PetscErrorCode FormDirichletPsiExact(Vec Uexact, ObsCtx *user) {
  PetscErrorCode ierr;
  int            i,j;
  double         **ag, **apsi, **auexact, dx, dy, x, y, r,
                 afree = 0.69797, A = 0.68026, B = 0.47152;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(user->da,&info); CHKERRQ(ierr);
  GridSpaces(&info,&dx,&dy);
  ierr = DMDAVecGetArray(user->da, user->g, &ag);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da, user->psi, &apsi);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da, Uexact, &auexact);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = -2.0 + j * dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = -2.0 + i * dx;
      r = PetscSqrtReal(x * x + y * y);
      if (r <= 1.0)
        apsi[j][i] = PetscSqrtReal(1.0 - r * r);
      else
        apsi[j][i] = -1.0;
      if (r <= afree)
        auexact[j][i] = apsi[j][i];  /* on the obstacle */
      else
        auexact[j][i] = - A * PetscLogReal(r) + B;   /* solves the laplace eqn */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1)
        ag[j][i] = auexact[j][i];
      else
        ag[j][i] = NAN;
    }
  }
  ierr = DMDAVecRestoreArray(user->da, user->g, &ag);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da, user->psi, &apsi);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da, Uexact, &auexact);CHKERRQ(ierr);
  return 0;
}

//  for call-back: tell SNESVI (variational inequality) that we want  psi <= u < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  PetscErrorCode ierr;
  ObsCtx *user;
  ierr = SNESGetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = VecCopy(user->psi,Xl);CHKERRQ(ierr);  /* u >= psi */
  ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
  return 0;
}

/* FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,ObsCtx *user) {
  PetscErrorCode ierr;
  int     i, j;
  double  dx, dy, hxhy, hyhx, uxx, uyy, **ag;
  GridSpaces(info,&dx,&dy);  hxhy = dx / dy;  hyhx = dy / dx;
  ierr = DMDAVecGetArray(info->da, user->g, &ag);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i] - ag[j][i];
      } else {
        uxx     = hyhx * (x[j][i-1] - 2.0 * x[j][i] + x[j][i+1]);
        uyy     = hxhy * (x[j-1][i] - 2.0 * x[j][i] + x[j+1][i]);
        f[j][i] = - uxx - uyy;
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, user->g, &ag);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  SNES           snes;
  Vec            u, uexact;   /* solution, exact solution */
  ObsCtx         user;
  double         error1,errorinf;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,NULL,help);

  /* setup */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      3,3,                       // override with -da_refine or -da_grid_x,_y
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &(user.da));CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
  ierr = DMSetUp(user.da); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da,-2.0,2.0,-2.0,2.0,-1.0,-1.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.g));CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.psi));CHKERRQ(ierr);
  ierr = FormDirichletPsiExact(uexact,&user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(user.da,
             (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* solve */
  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

  /* compare to exact */
  ierr = VecAXPY(u,-1.0,uexact);CHKERRQ(ierr); /* u <- u - uexact */
  ierr = VecNorm(u,NORM_1,&error1);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  error1 /= (double)info.mx * (double)info.my;
  ierr = VecNorm(u,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "errors on %3d x %3d grid: av |u-uexact| = %.3e, |u-uexact|_inf = %.3e\n",
      info.mx,info.my,error1,errorinf);CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  VecDestroy(&(user.psi));  VecDestroy(&(user.g));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}

