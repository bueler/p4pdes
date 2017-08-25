static const char help[] = "Solves obstacle problem in 2D as a variational\n\
inequality.  An elliptic problem with solution  u  constrained to be above a\n\
given function  psi.  Exact solution is known.  Because of the constraint,\n\
the problem is nonlinear.\n";

/*
parallel runs, spatial refinement, robust PC:
for M in 0 1 2 3 4 5 6; do mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu; done

grid sequencing really worthwhile:
    timer mpiexec -n 4 ./obstacle -snes_monitor -snes_converged_reason -snes_grid_sequence 8
    real 10.82
versus 180 seconds

check it out:
$ timer ./obstacle -da_refine 8 -pc_type ilu
errors on 513 x 513 grid: av |u-uexact| = 2.141e-06, |u-uexact|_inf = 1.950e-05
real 178.07
$ timer ./obstacle -da_refine 8 -pc_type mg
errors on 513 x 513 grid: av |u-uexact| = 2.137e-06, |u-uexact|_inf = 1.950e-05
real 38.58
$ timer ./obstacle -snes_grid_sequence 8 -pc_type ilu
errors on 513 x 513 grid: av |u-uexact| = 2.137e-06, |u-uexact|_inf = 1.950e-05
real 9.81
$ timer ./obstacle -snes_grid_sequence 8 -pc_type mg
errors on 513 x 513 grid: av |u-uexact| = 2.137e-06, |u-uexact|_inf = 1.950e-05
real 2.65

and then one can increase the dimension by 64 times!!!!:
$ timer ./obstacle -snes_grid_sequence 9 -pc_type mg
errors on 1025 x 1025 grid: av |u-uexact| = 7.130e-07, |u-uexact|_inf = 6.908e-06
real 11.38
$ timer ./obstacle -snes_grid_sequence 10 -pc_type mg
errors on 2049 x 2049 grid: av |u-uexact| = 2.243e-07, |u-uexact|_inf = 1.828e-06
real 38.63
$ timer ./obstacle -snes_grid_sequence 11 -pc_type mg
errors on 4097 x 4097 grid: av |u-uexact| = 1.231e-07, |u-uexact|_inf = 7.511e-07
real 147.49

one can adjust the smallest grid, which is needed in parallel to correctly cut-up
the coarsest grid, but starting small seems good:
$ timer ./obstacle -snes_converged_reason -pc_type mg -snes_grid_sequence 9
                  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
                Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
...
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
errors on 1025 x 1025 grid: av |u-uexact| = 7.130e-07, |u-uexact|_inf = 6.908e-06
real 11.39
$ timer ./obstacle -da_grid_x 33 -da_grid_y 33 -snes_converged_reason -pc_type mg -snes_grid_sequence 5
          Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
...
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
errors on 1025 x 1025 grid: av |u-uexact| = 7.130e-07, |u-uexact|_inf = 6.908e-06
real 12.16
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
  DM             da, da_after;
  SNES           snes;
  Vec            u_initial, u, u_exact;   /* solution, exact solution */
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
  ierr = DMCreateGlobalVector(da,&u_initial);CHKERRQ(ierr);
  ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u_initial);CHKERRQ(ierr);
  ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);

  /* compare to exact */
  ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
  ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
  ierr = FormUExact(&info,u_exact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr); /* u <- u - u_exact */
  ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_1,&error1); CHKERRQ(ierr);
  error1 /= (double)info.mx * (double)info.my;
  ierr = VecNorm(u,NORM_INFINITY,&errorinf); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "errors on %3d x %3d grid: av |u-uexact| = %.3e, |u-uexact|_inf = %.3e\n",
      info.mx,info.my,error1,errorinf); CHKERRQ(ierr);

  SNESDestroy(&snes);
  return PetscFinalize();
}

