static char help[] = "A structured-grid Poisson problem with DMDA+SNES.\n"
"Solves  -nabla^2 u = f  by putting it in form  F(u) = -nabla^2 u - f.\n"
"Multigrid-capable because the call-back works for the grid it is given.\n\n";

/* best solve I can get given the memory limitations of my machine:
PETSC_ARCH=linux-c-opt

$ timer mpiexec -n 4 ./fish2 -da_refine 9 -ksp_type cg -pc_type mg -ksp_converged_reason -ksp_rtol 1.0e-12
  Linear solve converged due to CONVERGED_RTOL iterations 6
on 4097 x 4097 grid:  error |u-uexact|_inf = 3.00114e-09
real 21.56

FIXME: why does it seg fault with -snes_fd_color, e.g.
  mpiexec -n 2 ./fish2 -da_refine 1 -ksp_type cg -pc_type mg -ksp_converged_reason -ksp_rtol 1.0e-12 -snes_fd_color
*/

#include <petsc.h>
#define COMM PETSC_COMM_WORLD

typedef struct {
    DM        da;
    Vec       b;
} FishCtx;

PetscErrorCode formExactRHS(DMDALocalInfo *info, Vec uexact, Vec b,
                            FishCtx* user) {
  PetscErrorCode ierr;
  const double hx = 1.0/(info->mx-1),  hy = 1.0/(info->my-1);
  int          i, j;
  double       x, y, f, **auexact, **ab;

  ierr = DMDAVecGetArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da, b, &ab);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = j * hy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = i * hx;
      auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
      if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
        ab[j][i] = 0.0;                    // on bdry the eqn is 1*u = 0
      } else {  // if not bdry; note  f = - (u_xx + u_yy)  where u is exact
        f = 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y)
                    + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) );
        ab[j][i] = hx * hy * f;
      }
    }
  }
  ierr = DMDAVecRestoreArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da, b, &ab);CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, FishCtx *user) {
  PetscErrorCode ierr;
  const double hx = 1.0/(info->mx-1),  hy = 1.0/(info->my-1);
  int          i, j;
  double       **ab;

  ierr = DMDAVecGetArray(user->da,user->b,&ab); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
              FF[j][i] = au[j][i];
          } else {
              FF[j][i] = 2*(hy/hx + hx/hy) * au[j][i]
                         - hy/hx * (au[j][i-1] + au[j][i+1])
                         - hx/hy * (au[j-1][i] + au[j+1][i])
                         - ab[j][i];
          }
      }
  }
  ierr = DMDAVecRestoreArray(user->da,user->b,&ab); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                 Mat J, Mat Jpre, FishCtx *user) {
    PetscErrorCode  ierr;
    const double hx = 1.0/(info->mx-1),  hy = 1.0/(info->my-1);
    int          i,j,ncols;
    double       v[5];
    MatStencil   col[5],row;

    for (j = info->ys; j < info->ys+info->ym; j++) {
        row.j = j;
        col[0].j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            row.i = i;
            col[0].i = i;
            ncols = 1;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                v[0] = 1.0;
            } else {
                v[0] = 2*(hy/hx + hx/hy);
                if (i-1 > 0) {
                col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = -hy/hx;  }
                if (i+1 < info->mx-1) {
                col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = -hy/hx;  }
                if (j-1 > 0) {
                col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
                if (j+1 < info->my-1) {
                col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
            }
            ierr = MatSetValuesStencil(Jpre,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  SNES           snes;
  Vec            u, uexact;
  FishCtx        user;
  DMDALocalInfo  info;
  double         errnorm;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(user.da)); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(user.da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.b));CHKERRQ(ierr);
  ierr = formExactRHS(&info,uexact,user.b,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(user.da,
             (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"on %d x %d grid:  error |u-uexact|_inf = %g\n",
           info.mx,info.my,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.b));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}

