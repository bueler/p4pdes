static char help[] =
"Structured-grid Poisson problem using DMDA+SNES.  Option prefix -f1_.\n"
"Solves  -u'' = f  by putting it in form  F(u) = -u'' - f.\n"
"Homogeneous Dirichlet boundary conditions on unit interval.\n"
"Multigrid-capable because call-backs discretize for the grid it is given.\n\n";

/* see study/mgstudy.sh for multigrid parameter study

view the basics with multigrid:
$ ./fish1 -da_refine 3 -pc_type mg -snes_view|less
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor -ksp_converged_reason

this makes sense and shows V-cycles:
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor -ksp_converged_reason -mg_levels_ksp_monitor|less

this additionally generate .m files with solutions at levels:
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor_solution ascii:u.m:ascii_matlab
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_3_ksp_monitor_solution ascii:errlevel3.m:ascii_matlab
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_2_ksp_monitor_solution ascii:errlevel2.m:ascii_matlab
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_1_ksp_monitor_solution ascii:errlevel1.m:ascii_matlab

because default -mg_coarse_ksp_type is preonly, without changing that we get nothing:
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_coarse_ksp_type cg -snes_monitor -ksp_converged_reason -mg_levels_ksp_monitor -mg_coarse_ksp_monitor|less
$ ./fish1 -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_coarse_ksp_type cg -mg_coarse_ksp_monitor_solution ascii:errcoarse.m:ascii_matlab

FD jacobian with coloring is actually faster (and it is clear what is going on):
$ timer ./fish1 -snes_monitor -da_refine 16
$ timer ./fish1 -snes_monitor -da_refine 16 -snes_fd_color

compare whether rediscretization happens at each level (former) or Galerkin grid-
transfer operators are used (latter)
$ ./fish1 -da_refine 4 -pc_type mg -snes_monitor
$ ./fish1 -da_refine 4 -pc_type mg -snes_monitor -pc_mg_galerkin

choose linear solver for coarse grid (default is preonly+lu):
$ ./fish1 -da_refine 4 -pc_type mg -mg_coarse_ksp_type cg -mg_coarse_pc_type jacobi -ksp_view|less

*/

#include <petsc.h>
#include "jacobians.c"
#define COMM PETSC_COMM_WORLD

typedef struct {
    DM        da;
    Vec       b;
} FishCtx;

PetscErrorCode formExactRHS(DMDALocalInfo *info, Vec uexact, Vec b,
                            FishCtx* user) {
  PetscErrorCode ierr;
  const double hx = 1.0/(info->mx-1);
  int          i;
  double       x, f, *auexact, *ab;

  ierr = DMDAVecGetArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da, b, &ab);CHKERRQ(ierr);
  for (i=info->xs; i<info->xs+info->xm; i++) {
      x = i * hx;
      auexact[i] = x*x * (1.0 - x*x);
      if (i==0 || i==info->mx-1) {
        ab[i] = 0.0;                    // on bdry the eqn is 1*u = 0
      } else {  // if not bdry; note  f = - u''  where u is exact
        f = 12.0 * x*x - 2.0;
        ab[i] = hx * hx * f;
      }
  }
  ierr = DMDAVecRestoreArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da, b, &ab);CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double *au,
                                 double *FF, FishCtx *user) {
    PetscErrorCode ierr;
    int          i;
    double       *ab;

    ierr = DMDAVecGetArray(user->da,user->b,&ab); CHKERRQ(ierr);
    for (i = info->xs; i < info->xs + info->xm; i++) {
        if (i==0 || i==info->mx-1) {
            FF[i] = au[i];
        } else {
            FF[i] = 2.0 * au[i] - au[i-1] - au[i+1] - ab[i];
        }
    }
    ierr = DMDAVecRestoreArray(user->da,user->b,&ab); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  SNES           snes;
  KSP            ksp;
  Vec            u, uexact;
  FishCtx        user;
  DMDALocalInfo  info;
  double         errnorm;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate1d(COMM, DM_BOUNDARY_NONE, 3, 1, 1, NULL, &(user.da)); CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.da); CHKERRQ(ierr);
  ierr = DMSetUp(user.da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.b));CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);
  ierr = formExactRHS(&info,uexact,user.b,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(user.da,
             (DMDASNESJacobian)Form1DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"on %d grid:  error |u-uexact|_inf = %g\n",
           info.mx,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.b));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}

