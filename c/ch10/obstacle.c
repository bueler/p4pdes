static const char help[] = "Solves obstacle problem in 2D as a variational\n\
inequality.  An elliptic problem with solution  u  constrained to be above a\n\
given function  psi.  Exact solution is known.  Because of the constraint,\n\
the problem is nonlinear.\n";

/*
parallel runs, spatial refinement, robust PC:
for M in 0 1 2 3 4 5 6; do mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu; done

grid sequencing really worthwhile; check out these serial runs:
$ timer ./obstacle -da_refine 8 -pc_type ilu
errors on 513 x 513 grid: av |u-uexact| = 2.055e-06, |u-uexact|_inf = 1.918e-05
real 143.77
$ timer ./obstacle -da_refine 8 -pc_type mg
errors on 513 x 513 grid: av |u-uexact| = 2.051e-06, |u-uexact|_inf = 1.918e-05
real 27.44
$ timer ./obstacle -snes_grid_sequence 8 -pc_type ilu
errors on 513 x 513 grid: av |u-uexact| = 2.051e-06, |u-uexact|_inf = 1.918e-05
real 7.90
$ timer ./obstacle -snes_grid_sequence 8 -pc_type mg
errors on 513 x 513 grid: av |u-uexact| = 2.051e-06, |u-uexact|_inf = 1.918e-05
real 1.91

and then one can increase the dimension by 64 times!!!! (still serial; level 12
would exceed memory on WORKSTATION):
$ timer ./obstacle -snes_grid_sequence 9 -pc_type mg
errors on 1025 x 1025 grid: av |u-uexact| = 6.266e-07, |u-uexact|_inf = 6.592e-06
real 8.18
$ timer ./obstacle -snes_grid_sequence 10 -pc_type mg
errors on 2049 x 2049 grid: av |u-uexact| = 1.379e-07, |u-uexact|_inf = 1.511e-06
real 29.15
$ timer ./obstacle -snes_grid_sequence 11 -pc_type mg
errors on 4097 x 4097 grid: av |u-uexact| = 3.657e-08, |u-uexact|_inf = 4.332e-07
real 111.59

one can adjust the smallest grid, which is needed in parallel to correctly cut-up
the coarsest grid, but starting small seems good:
$ timer ./obstacle -snes_converged_reason -pc_type mg -snes_grid_sequence 9
                  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
                Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
...
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
errors on 1025 x 1025 grid: av |u-uexact| = 6.266e-07, |u-uexact|_inf = 6.592e-06
real 8.33
$ timer ./obstacle -da_grid_x 33 -da_grid_y 33 -snes_converged_reason -pc_type mg -snes_grid_sequence 5
          Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 5
        Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
      Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
  Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 3
errors on 1025 x 1025 grid: av |u-uexact| = 6.266e-07, |u-uexact|_inf = 6.592e-06
real 8.31

parallel versions of above two runs:
STALLS: $ timer mpiexec -n 4 ./obstacle -snes_converged_reason -pc_type mg -snes_grid_sequence 9
SUCCEEDS IN 6.18 secs: $ timer mpiexec -n 4 ./obstacle -da_grid_x 33 -da_grid_y 33 -snes_converged_reason -pc_type mg -snes_grid_sequence 5
*/

#include <petsc.h>
#include "../ch6/poissonfunctions.h"

// z = psi(x,y) is the obstacle
double psi(double x, double y) {
    double  rr = x * x + y * y;
    return (rr <= 1.0) ? PetscSqrtReal(1.0 - rr) : -1.0;
}

/* To get exact solution see
    https://github.com/bueler/fem-code-challenge/blob/master/obstacleDOC.pdf

In summary, solve a 1D radial free-boundary problem on interval 0 < r < 2 with
obstacle

    psi(r) =  / sqrt(1 - r^2),  r < 1
              \ -1,             otherwise

and Laplace equation where u(r) > psi(r),

    u''(r) + r^-1 u'(r) = 0

with (free) boundary conditions

    u(a) = psi(a),  u'(a) = psi'(a),  u(2) = 0

The solution is  u(r) = - A log(r) + B   on the interval  a < r < 2.  The
boundary conditions can then be reduced to a root-finding problem for a:

    a^2 (log(2) - log(a)) = 1 - a^2

Solved via Matlab/Octave:
>> f = @(a) a.^2 .* (log(2)-log(a)) - 1 + a.^2;
>> a = 0.2:0.001:1.0; plot(a,f(a)), grid on
>> a = fzero(f,0.7)
a =    0.697965148223374
>> f(a)
ans = 1.49880108324396e-15
>> A = a^2*(1-a^2)^(-0.5),  B = A*log(2)
A =    0.680259411891719
B =    0.471519893402112

These constants appear in the C function below.                    */
double u_exact(double x, double y) {
    const double afree = 0.697965148223374,
                 A     = 0.680259411891719,
                 B     = 0.471519893402112;
    double  r;
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

  user.cx = 1.0;
  user.cy = 1.0;
  user.cz = 1.0;
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
             (DMDASNESFunction)Poisson2DFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Poisson2DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // petsc bug: I get error inside VINEWTONRSLS if I use DMGetGlobalVector() here
  //   PETSC ERROR: Clearing DM of global vectors that has a global vector obtained with DMGetGlobalVector()
  //   ...
  //   PETSC ERROR: #2 DMSetVI() line 224 in /home/ed/petsc/src/snes/impls/vi/rs/virs.c
  //   PETSC ERROR: #3 SNESSolve_VINEWTONRSLS() line 431 in /home/ed/petsc/src/snes/impls/vi/rs/virs.c
  ierr = DMCreateGlobalVector(da,&u_initial);CHKERRQ(ierr);
  // initial iterate has u=g on boundary and u=0 in interior
  ierr = InitialState(da, ZEROS, PETSC_TRUE, u_initial, &user); CHKERRQ(ierr);

  /* solve */
  ierr = SNESSolve(snes,NULL,u_initial);CHKERRQ(ierr);
  ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);

  /* compare to exact */
  ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr); /* do not destroy u */
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

