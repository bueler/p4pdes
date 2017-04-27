static char help[] =
"Structured-grid Poisson problem in 2D using DMDA+SNES.\n"
"Solves  -nabla^2 u = f  by putting it in form  F(u) = -nabla^2 u - f.\n"
"Homogeneous Dirichlet boundary conditions on unit square.\n"
"Multigrid-capable because call-backs discretize for the supplied grid.\n\n";

/* see study/mgstudy.sh for multigrid parameter study

this makes sense and the latter shows V-cycles:
$ ./fish2 -da_refine 3 -pc_type mg -snes_monitor -ksp_converged_reason
$ ./fish2 -da_refine 3 -pc_type mg -snes_monitor -ksp_converged_reason -mg_levels_ksp_monitor|less

multigrid seg faults in parallel with -snes_fd_color, but -pc_mg_galerkin fixes it:
  bad:   mpiexec -n 2 ./fish2 -da_refine 4 -pc_type mg -snes_fd_color
  good:  mpiexec -n 2 ./fish2 -da_refine 4 -pc_type mg -snes_fd_color -pc_mg_galerkin

compare whether rediscretization happens at each level (former) or Galerkin grid-
transfer operators are used (latter)
$ ./fish2 -da_refine 4 -pc_type mg -snes_monitor
$ ./fish2 -da_refine 4 -pc_type mg -snes_monitor -pc_mg_galerkin

choose linear solver for coarse grid (default is preonly+lu):
$ ./fish2 -da_refine 4 -pc_type mg -mg_coarse_ksp_type cg -mg_coarse_pc_type jacobi -ksp_view|less

to generate classical jacobi/gauss-seidel results, add
   PetscViewer viewer;
   PetscViewerASCIIOpen(COMM,"rhs.m",&viewer);
   PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
   VecView(user.f,viewer);
after setting user.f, and then do:
$ ./fish2 -da_refine 1 -snes_monitor -ksp_monitor -snes_max_it 1 -ksp_type richardson -pc_type jacobi|sor
with e.g. -ksp_monitor_solution :foo.m:ascii_matlab
*/

#include <petsc.h>
#include "jacobians.c"
#define COMM PETSC_COMM_WORLD

typedef struct {
    Vec       f;
} FishCtx;

PetscErrorCode formExactRHS(DMDALocalInfo *info, Vec uexact, Vec f,
                            FishCtx* user) {
  PetscErrorCode ierr;
  int          i, j;
  double       xymin[2], xymax[2], hx, hy, x, y, **auexact, **af;
  ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
  hx = (xymax[0] - xymin[0]) / (info->mx - 1);
  hy = (xymax[1] - xymin[1]) / (info->my - 1);
  ierr = DMDAVecGetArray(info->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da, f, &af);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = xymin[1] + j * hy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = xymin[0] + i * hx;
      auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
      if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
        af[j][i] = 0.0;                    // on bdry the eqn is 1*u = 0
      } else {  // if not bdry; note  f = - (u_xx + u_yy)  where u is exact
        af[j][i] = 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y)
                       + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) );
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(info->da, f, &af);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, FishCtx *user) {
    PetscErrorCode ierr;
    int          i, j;
    double       hx, hy, xymin[2], xymax[2], ue, uw, un, us, **af;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da,user->f,&af); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                FF[j][i] = au[j][i];
            } else {
                ue = (i+1 == info->mx-1) ? 0.0 : au[j][i+1];
                uw = (i-1 == 0)          ? 0.0 : au[j][i-1];
                un = (j+1 == info->my-1) ? 0.0 : au[j+1][i];
                us = (j-1 == 0)          ? 0.0 : au[j-1][i];
                FF[j][i] = 2.0 * (hy/hx + hx/hy) * au[j][i]
                           - hy/hx * (uw + ue)
                           - hx/hy * (us + un)
                           - hx * hy * af[j][i];
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da,user->f,&af); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  KSP            ksp;
  Vec            u, uexact;
  FishCtx        user;
  DMDALocalInfo  info;
  double         errinf,err2h;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_STAR,
               3,3,PETSC_DECIDE,PETSC_DECIDE,
               1,1,NULL,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.f));CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = formExactRHS(&info,uexact,user.f,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&err2h); CHKERRQ(ierr);
  err2h /= PetscSqrtReal((double)(info.mx-1)*(info.my-1));// like continuous L2
  ierr = PetscPrintf(COMM,
           "on %d x %d grid:  error |u-uexact|_inf = %g, |...|_h = %.2e\n",
           info.mx,info.my,errinf,err2h); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.f));
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

