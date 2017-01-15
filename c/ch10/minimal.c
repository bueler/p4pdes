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
} MinimalCtx;

double GG(double H, double x) {
    return 2.0 * H * (x < 0.5 ? x : (1.0 - x));
}

// FIXME   make it work in parallel

// FIXME   add RHS f(x,y) and manufacture a solution

// FIXME   write computeArea()

// the diffusivity as a function of  z = |nabla u|^2
double DD(double z) { 
    return 1.0 / sqrt(1.0 + z);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, MinimalCtx *user) {
    PetscErrorCode ierr;
    int          i, j;
    double       xymin[2], xymax[2], hx, hy, x,
                 dux, duy, De, Dw, Dn, Ds, **av;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (j==0) {
                FF[j][i] = au[j][i] - GG(user->H,x);
            } else if (i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i];
            } else {
                if (i+1 == info->mx-1) {
                    ue 
                }
FIXME  too sleepy to decide how to do this
                ue = (i+1 == info->mx-1) ? 0.0 : au[j][i+1];
                uw = (i-1 == 0)          ? 0.0 : au[j][i-1];
                un = (j+1 == info->my-1) ? 0.0 : au[j+1][i];
                us = (j-1 == 0)          ? GG(user->H,xs) : au[j-1][i];
                if (user->laplace) {
                    De = 1.0;  Dw = 1.0;
                    Dn = 1.0;  Ds = 1.0;
                } else {
                    // gradient of u squared at east point  (i+1/2,j):
                    dux = (av[j][i+1] - av[j][i]) / hx;
                    duy = (av[j+1][i] + av[j+1][i+1] - av[j-1][i] - av[j-1][i+1]) / (4.0 * hy);
                    De = DD(dux * dux + duy * duy);
                    // ...                   at west point  (i-1/2,j):
                    dux = (av[j][i] - av[j][i-1]) / hx;
                    duy = (av[j+1][i-1] + av[j+1][i] - av[j-1][i-1] - av[j-1][i]) / (4.0 * hy);
                    Dw = DD(dux * dux + duy * duy);
                    // ...                  at north point  (i,j+1/2):
                    dux = (av[j][i+1] + av[j+1][i+1] - av[j][i-1] - av[j+1][i-1]) / (4.0 * hx);
                    duy = (av[j+1][i] - av[j][i]) / hy;
                    Dn = DD(dux * dux + duy * duy);
                    // ...                  at south point  (i,j-1/2):
                    dux = (av[j][i+1] + av[j-1][i+1] - av[j][i-1] - av[j-1][i-1]) / (4.0 * hx);
                    duy = (av[j][i] - av[j-1][i]) / hy;
                    Ds = DD(dux * dux + duy * duy);
                }
                // evaluate residual
                FF[j][i] = - hy/hx * (   De * (av[j][i+1] - av[j][i])
                                       - Dw * (av[j][i] - av[j][i-1]) )
                           - hx/hy * (   Dn * (av[j+1][i] - av[j][i])
                                       - Ds * (av[j][i] - av[j-1][i]) );
            }
        }
    }
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

    VecDestroy(&u);  SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

