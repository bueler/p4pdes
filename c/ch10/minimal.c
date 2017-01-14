static char help[] =
"Structured-grid minimal surface equation in 2D using DMDA+SNES.\n"
"Option prefix mse_.\n"
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
    double    H;       // height of tent along y=0 boundary
    PetscBool laplace; // solve Laplace equation instead of minimal surface
    Vec       ucopy;   // workspace
} MinimalCtx;

// FIXME   write computeArea()

// the diffusivity as a function of  z = |nabla u|^2
double DD(double z) { 
    return 1.0 / sqrt(1.0 + z);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, MinimalCtx *user) {
    PetscErrorCode ierr;
    int          i, j;
    double       xymin[2], xymax[2], hx, hy, x, g,
                 ux, uy, De, Dw, Dn, Ds, **av;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da,user->ucopy,&av); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (j==0) {
                x = xymin[0] + i * hx;
                g = 2.0 * user->H * (x < 0.5 ? x : (1.0 - x));
                FF[j][i] = au[j][i] - g;
                av[j][i] = g;
            } else if (i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i];
                av[j][i] = 0.0;
            } else {
                av[j][i] = au[j][i];
            }
        }
    }
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (i>0 && j>0 && i<info->mx-1 && j<info->my-1) {
                if (user->laplace) {
                    De = 1.0;  Dw = 1.0;
                    Dn = 1.0;  Ds = 1.0;
                } else {
                    // gradient of u squared at east point  (i+1/2,j):
                    ux = (av[j][i+1] - av[j][i]) / hx;
                    uy = (av[j+1][i] + av[j+1][i+1] - av[j-1][i] - av[j-1][i+1]) / (4.0 * hy);
                    De = DD(ux * ux + uy * uy);
                    // ...                   at west point  (i-1/2,j):
                    ux = (av[j][i] - av[j][i-1]) / hx;
                    uy = (av[j+1][i-1] + av[j+1][i] - av[j-1][i-1] - av[j-1][i]) / (4.0 * hy);
                    Dw = DD(ux * ux + uy * uy);
                    // ...                  at north point  (i,j+1/2):
                    ux = (av[j][i+1] + av[j+1][i+1] - av[j][i-1] - av[j+1][i-1]) / (4.0 * hx);
                    uy = (av[j+1][i] - av[j][i]) / hy;
                    Dn = DD(ux * ux + uy * uy);
                    // ...                  at south point  (i,j-1/2):
                    ux = (av[j][i+1] + av[j-1][i+1] - av[j][i-1] - av[j-1][i-1]) / (4.0 * hx);
                    uy = (av[j][i] - av[j-1][i]) / hy;
                    Ds = DD(ux * ux + uy * uy);
                }
                // evaluate residual
                FF[j][i] = - hy/hx * (   De * (av[j][i+1] - av[j][i])
                                       - Dw * (av[j][i] - av[j][i-1]) )
                           - hx/hy * (   Dn * (av[j+1][i] - av[j][i])
                                       - Ds * (av[j][i] - av[j-1][i]) );
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da,user->ucopy,&av); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u;
    MinimalCtx     user;
    DMDALocalInfo  info;

    PetscInitialize(&argc,&argv,NULL,help);

    user.H = 1.0;
    user.laplace = PETSC_FALSE;
    ierr = PetscOptionsBegin(COMM,"mse_","minimal surface equation solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-H","tent height",
                            "minimal.c",user.H,&(user.H),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-laplace","solve Laplace equation instead of minimal surface",
                            "minimal.c",user.laplace,&(user.laplace),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    //FIXME add option between pure laplacian and minimal surface equations, with
    //      comparison of surface areas for same boundary conditions, via objective

    ierr = DMDACreate2d(COMM,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,  // contrast with fish2
                        3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                        &da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);
    ierr = VecDuplicate(u,&(user.ucopy));CHKERRQ(ierr);

    ierr = SNESCreate(COMM,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    // this is the Jacobian of the Poisson equation, thus only approximate;
    // consider using -snes_mf_operator
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(COMM,"done on %d x %d grid ...\n",
                       info.mx,info.my); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&(user.ucopy));
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

