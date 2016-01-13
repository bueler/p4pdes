static char help[] = "A structured-grid Poisson problem with DMDA+SNES.\n"
"Solves  -nabla^2 u = f  by putting it in form  F(u) = -nabla^2 u - f.\n"
"Multigrid-capable because the call-back works for the grid it is given.\n\n";

#include <petsc.h>
#define COMM PETSC_COMM_WORLD

typedef struct {
    DM        da;
    Vec       b;
} FishCtx;

PetscErrorCode formExactRHS(DMDALocalInfo *info, Vec uexact, Vec b, FishCtx* user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0/(info->mx-1),  hy = 1.0/(info->my-1);
  PetscInt        i, j;
  PetscReal       x, y, x2, y2, f, **auexact, **ab;

  ierr = DMDAVecGetArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da, b, &ab);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = j * hy;
    y2 = y * y;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = i * hx;
      x2 = x * x;
      auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
      if ( (i>0) && (i<info->mx-1) && (j>0) && (j<info->my-1) ) { // if not bdry
        // f = - (u_xx + u_yy)  where u is exact
        f = 2.0 * ( (1.0 - 6.0*x2) * y2 * (1.0 - y2)
                    + (1.0 - 6.0*y2) * x2 * (1.0 - x2) );
        ab[j][i] = hx * hy * f;
      } else {
        ab[j][i] = 0.0;                          // on bdry the eqn is 1*u = 0
      }
    }
  }
  ierr = DMDAVecRestoreArray(user->da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da, b, &ab);CHKERRQ(ierr);
  return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, FishCtx *user) {
  PetscErrorCode ierr;
  const PetscReal hx = 1.0/(info->mx-1),  hy = 1.0/(info->my-1);
  PetscInt        i, j;
  PetscReal       **ab;

  // compute residual FF[j][i] for each node (x_i,y_j)
  ierr = DMDAVecGetArray(user->da,user->b,&ab); CHKERRQ(ierr);
  for (j = info->ys; j < info->ys + info->ym; j++) {
      for (i = info->xs; i < info->xs + info->xm; i++) {
          if ( (i==0) || (i==info->mx-1) || (j==0) || (j==info->my-1) ) {
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


int main(int argc,char **argv) {
  PetscErrorCode ierr;
  SNES           snes;
  Vec            u, uexact;
  FishCtx        user;
  DMDALocalInfo  info;
  PetscReal      errnorm;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
               -9,-9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&(user.da)); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(user.da,&info); CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.b));CHKERRQ(ierr);
  ierr = formExactRHS(&info,uexact,user.b,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,user.da); CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(user.da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
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

