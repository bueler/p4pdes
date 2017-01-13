static char help[] =
"Structured-grid minimal surface equation in 2D using DMDA+SNES.\n"
"Option prefix min_.\n"
"Solves\n"
"            /         nabla u         \\ \n"
"  - nabla . | ----------------------- | = 0\n"
"            \\  sqrt(1 + |nabla u|^2)  / \n"
"subject to Dirichlet boundary conditions  u = g  on boundary of unit square.\n"
"Allows re-use of Jacobian (Laplacian) from fish2 as preconditioner.\n"
"Multigrid-capable.\n\n";

#include <petsc.h>
#include "jacobians.c"
#define COMM PETSC_COMM_WORLD

typedef struct {
    Vec       g;
} MinimalCtx;

PetscErrorCode formDirichlet(DMDALocalInfo *info, Vec uexact, Vec g,
                             MinimalCtx* user) {
  PetscErrorCode ierr;
  int          i, j;
  double       xymin[2], xymax[2], hx, hy, x, y, **ag;
  ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
  hx = (xymax[0] - xymin[0]) / (info->mx - 1);
  hy = (xymax[1] - xymin[1]) / (info->my - 1);
  ierr = DMDAVecGetArray(info->da, g, &ag);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    y = xymin[1] + j * hy;
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = xymin[0] + i * hx;
      auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
      if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
FIXME
        af[j][i] = 0.0;
      } else {  // if not bdry then invalidate
        af[j][i] = NAN;
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, g, &ag);CHKERRQ(ierr);
  return 0;
}

FIXME   write FormObjectiveLocal()

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, MinimalCtx *user) {
    PetscErrorCode ierr;
    int          i, j;
    double       hx, hy, xymin[2], xymax[2], **af;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da,user->g,&ag); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                FF[j][i] = au[j][i] - ag[j][i];
            } else {
FIXME
                FF[j][i] = 2.0 * (hy/hx + hx/hy) * au[j][i]
                           - hy/hx * (au[j][i-1] + au[j][i+1])
                           - hx/hy * (au[j-1][i] + au[j+1][i])
                           - hx * hy * af[j][i];
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da,user->g,&ag); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da;
  SNES           snes;
  KSP            ksp;
  Vec            u;
  MinimalCtx     user;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(COMM,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_BOX,  // contrast with fish2
               3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.g));CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  ierr = formDirichlet(&info,uexact,user.g,&user); CHKERRQ(ierr);

  ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
FIXME   DMDASNESSetObjectiveLocal()
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);

  // this is the Jacobian of the Poisson equation, thus only approximate;
  // consider using -snes_mf_operator
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);

  ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
  ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  ierr = PetscPrintf(COMM,"done on %d x %d grid ...\n",
           info.mx,info.my); CHKERRQ(ierr);

  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&(user.g));
  SNESDestroy(&snes);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

