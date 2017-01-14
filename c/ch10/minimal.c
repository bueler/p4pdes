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
    Vec       g;  // Dirichlet boundary values (invalid in interior)
    double    H;  // height of tent along y=0 boundary
} MinimalCtx;

PetscErrorCode formDirichlet(DMDALocalInfo *info, Vec uexact, Vec g,
                             MinimalCtx* user) {
  PetscErrorCode ierr;
  int          i, j;
  double       xymin[2], xymax[2], hx, x, **ag;
  ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
  hx = (xymax[0] - xymin[0]) / (info->mx - 1);
  ierr = DMDAVecGetArray(info->da, g, &ag);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x = xymin[0] + i * hx;
      if (j==0) {
        ag[j][i] = 2.0 * user->H * (x < 0.5 ? x : (1.0 - x));
      } else if (i==0 || i==info->mx-1 || j==info->my-1) {
        ag[j][i] = 0.0;
      } else {  // if not bdry then invalidate
        ag[j][i] = NAN;
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, g, &ag);CHKERRQ(ierr);
  return 0;
}

// FIXME   write computeArea()

// the diffusivity as a function of  z = |nabla u|^2
double DD(double z) { 
    return 1.0 / sqrt(1.0 + z);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, MinimalCtx *user) {
    PetscErrorCode ierr;
    int          i, j;
    double       hx, hy, ux, uy, De, Dw, Dn, Ds, xymin[2], xymax[2], **ag;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da,user->g,&ag); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                FF[j][i] = au[j][i] - ag[j][i];
            } else {
                // gradient of u squared at east point  (i+1/2,j):
                ux = (au[j][i+1] - au[j][i]) / hx;
                uy = (au[j+1][i] + au[j+1][i+1] - au[j-1][i] - au[j-1][i+1]) / (4.0 * hy);
                De = DD(ux * ux + uy * uy);
FIXME
                //                       at west point  (i-1/2,j):
                ux = (au[j][i] - au[j][i-1]) / hx;
                uy = (au[j+1][i-1] + au[j+1][i] - au[j-1][i-1] - au[j-1][i]) / (4.0 * hy);
                GSw = ux * ux + uy * uy;
                //                      at north point  (i,j+1/2):
                ux = (au[j][i+1] + au[j+1][i+1] - au[j][i-1] - au[j+1][i-1]) / (4.0 * hx);
                uy = (au[j+1][i] - au[j][i]) / hy;
                GSn = ux * ux + uy * uy;
                //                      at south point  (i,j-1/2):
                //FIXME
                GSs = ux * ux + uy * uy;
                FF[j][i] = - hy/hx * ( De * (au[j][i+1] + au[j][i])
                                       - Dw * (au[j][i] + au[j][i-1]) )
                           - hx/hy * ( Dn * (au[j+1][i] + au[j][i])
                                       - Ds * (au[j][i] + au[j-1][i]) );
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

FIXME add option between pure laplacian and minimal surface equations, with
      comparison of surface areas for same boundary conditions, via objective

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

