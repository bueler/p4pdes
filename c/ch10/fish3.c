static char help[] =
"Structured-grid Poisson problem in 3D using DMDA+SNES.  Compare fish2.c.\n\n";

/* see study/mgstudy.sh for multigrid parameter study */

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
    const double hx = 1.0/(info->mx-1),
                 hy = 1.0/(info->my-1),
                 hz = 1.0/(info->mz-1),
                 h = pow(hx*hy*hz,1.0/3.0);
    int          i, j, k;
    double       x, y, z, f, ***auexact, ***ab,
                 aa, bb, cc, ddaa, ddbb, ddcc;
    ierr = DMDAVecGetArray(user->da, uexact, &auexact);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(user->da, b, &ab);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = k * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = j * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = i * hx;
                aa = x*x * (1.0 - x*x);
                bb = y*y * (y*y - 1.0);
                cc = z*z * (z*z - 1.0);
                auexact[k][j][i] = aa * bb * cc;
                if (   i==0 || i==info->mx-1
                    || j==0 || j==info->my-1
                    || k==0 || k==info->mz-1) {
                    ab[k][j][i] = 0.0;
                } else {  // if not bdry; note  f = - (u_xx + u_yy + u_zz)  where u is exact
                    ddaa = 2.0 * (1.0 - 6.0 * x*x);
                    ddbb = 2.0 * (6.0 * y*y - 1.0);
                    ddcc = 2.0 * (6.0 * z*z - 1.0);
                    f = - (ddaa * bb * cc + aa * ddbb * cc + aa * bb * ddcc);
                    ab[k][j][i] = h * h * f;
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(user->da, uexact, &auexact);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(user->da, b, &ab);CHKERRQ(ierr);
    return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double ***au,
                                 double ***FF, FishCtx *user) {
    PetscErrorCode ierr;
    const double hx = 1.0/(info->mx-1),
                 hy = 1.0/(info->my-1),
                 hz = 1.0/(info->mz-1),
                 h = pow(hx*hy*hz,1.0/3.0),
                 cx = h*h / (hx*hx),
                 cy = h*h / (hy*hy),
                 cz = h*h / (hz*hz);
    int          i, j, k;
    double       ***ab;

    ierr = DMDAVecGetArray(user->da,user->b,&ab); CHKERRQ(ierr);
    for (k = info->zs; k < info->zs + info->zm; k++) {
        for (j = info->ys; j < info->ys + info->ym; j++) {
            for (i = info->xs; i < info->xs + info->xm; i++) {
                if (   i==0 || i==info->mx-1
                    || j==0 || j==info->my-1
                    || k==0 || k==info->mz-1) {
                    FF[k][j][i] = au[k][j][i];
                } else {
                    FF[k][j][i] = 2.0 * (cx + cy + cz) * au[k][j][i]
                                  - cx * (au[k][j][i-1] + au[k][j][i+1])
                                  - cy * (au[k][j-1][i] + au[k][j+1][i])
                                  - cz * (au[k-1][j][i] + au[k+1][j][i])
                                  - ab[k][j][i];
                }
            }
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

  ierr = DMDACreate3d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_STAR,
               3,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
               1,1,
               NULL,NULL,NULL,&(user.da)); CHKERRQ(ierr);
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
             (DMDASNESJacobian)Form3DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = VecSet(u,0.0); CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"on %d x %d x %d grid:  error |u-uexact|_inf = %g\n",
           info.mx,info.my,info.mz,errnorm); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.b));
  SNESDestroy(&snes);  DMDestroy(&(user.da));
  PetscFinalize();
  return 0;
}

