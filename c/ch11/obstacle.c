static const char help[] = "Solves obstacle problem in 2D as a variational\n\
inequality.  An elliptic problem with solution  u  constrained to be above a\n\
given function  psi.  Exact solution is known.  Because of the constraint,\n\
the problem is nonlinear.\n";

/*
parallel runs, spatial refinement, robust PC:
for M in 0 1 2 3 4 5 6; do mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu; done

multigrid gets finer (use --with-debugging=0):
for M in 0 1 2 3 4 5 6 7 8; do mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu; done
*/

#include <petsc.h>
#include "../ch6/poissonfunctions.h"

// z = psi(x,y) is the obstacle
double psi(double x, double y) {
    double  r = PetscSqrtReal(x * x + y * y);
    return (r <= 1.0) ? PetscSqrtReal(1.0 - r * r) : -1.0;
}

// exact solution
double u_exact(double x, double y) {
    // FIXME: precision limited because these are given only to 10^-5
    double  r, afree = 0.69797, A = 0.68026, B = 0.47152;
    r = PetscSqrtReal(x * x + y * y);
    return (r <= afree) ? psi(x,y)  // on the obstacle
                        : - A * PetscLogReal(r) + B; // solves laplace eqn
}

// boundary conditions from exact solution
double g_fcn(double x, double y, double z, void *ctx) {
    return u_exact(x,y);
}

double zero(double x, double y, double z, void *ctx) {
    return 0.0;
}

PetscErrorCode FormUExact(DMDALocalInfo *info, Vec u) {
  PetscErrorCode ierr;
  int     i,j;
  double  **au, dx, dy, x, y;
  dx = 4.0 / (PetscReal)(info->mx-1);
  dy = 4.0 / (PetscReal)(info->my-1);
  ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = -2.0 + j * dy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = -2.0 + i * dx;
      au[j][i] = u_exact(x,y);
    }
  }
  ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
  return 0;
}

//  for call-back: tell SNESVI (variational inequality) that we want  psi <= u < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  PetscErrorCode ierr;
  DM            da;
  DMDALocalInfo info;
  int           i, j;
  double        **aXl, dx, dy, x, y;
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = 4.0 / (PetscReal)(info.mx-1);
  dy = 4.0 / (PetscReal)(info.my-1);
  ierr = DMDAVecGetArray(da, Xl, &aXl);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = -2.0 + j * dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = -2.0 + i * dx;
      aXl[j][i] = psi(x,y);
    }
  }
  ierr = DMDAVecRestoreArray(da, Xl, &aXl);CHKERRQ(ierr);
  ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  Vec            u, uexact;   /* solution, exact solution */
  PoissonCtx     user;
  double         error1,errorinf;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      3,3,                       // override with -da_refine or -da_grid_x,_y
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,-1.0,-1.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  user.g_bdry = &g_fcn;
  user.f_rhs = &zero;
  user.addctx = NULL;
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);

  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);
  
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)Form2DFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* solve */
  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

  /* compare to exact */
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = FormUExact(&info,uexact);CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact);CHKERRQ(ierr); /* u <- u - uexact */
  ierr = VecNorm(u,NORM_1,&error1);CHKERRQ(ierr);
  error1 /= (double)info.mx * (double)info.my;
  ierr = VecNorm(u,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "errors on %3d x %3d grid: av |u-uexact| = %.3e, |u-uexact|_inf = %.3e\n",
      info.mx,info.my,error1,errorinf);CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

