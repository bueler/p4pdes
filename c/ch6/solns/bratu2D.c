static char help[] =
"Solve nonlinear Liouville-Bratu equation in 2D on a structured-grid.  Option prefix lb_.\n"
"Solves\n"
"  - nabla^2 u + lambda ^u = f(x,y)\n"
"on the unit square [0,1]x[0,1] subject to zero Dirichlet boundary conditions.\n"
"Critical value occurs about at lambda = 6.808.\n";

/* compare:
timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8
timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8 -snes_fd_color
timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8 -snes_mf_operator
timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -snes_grid_sequence 8
*/

#include <petsc.h>
#include "../poissonfunctions.h"

typedef struct {
    double    lambda;
} BratuCtx;

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PoissonCtx *user) {
    BratuCtx   *bctx = (BratuCtx*)(user->addctx);
    int        i, j;
    double     hx, hy, hxhy, hyhx;
    hx = 1.0 / (double)(info->mx - 1);
    hy = 1.0 / (double)(info->my - 1);
    hxhy = hx / hy;
    hyhx = hy / hx;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (j==0 || i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i];
            } else {
                FF[j][i] =   hyhx * (2.0 * au[j][i] - au[j][i-1] - au[j][i+1])
                           + hxhy * (2.0 * au[j][i] - au[j-1][i] - au[j+1][i])
                           - hx * hy * bctx->lambda * PetscExpScalar(au[j][i]);
            }
        }
    }
    return 0;
}

// FIXME grab NonlinearGS() from branch add-ngs-to-minimal

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial;
    PoissonCtx     user;
    BratuCtx       bctx;
    DMDALocalInfo  info;

    PetscInitialize(&argc,&argv,NULL,help);
    user.Lx = 1.0;
    user.Ly = 1.0;
    user.Lz = 1.0;
    user.cx = 1.0;
    user.cy = 1.0;
    user.cz = 1.0;
    bctx.lambda = 1.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu2D.c",bctx.lambda,&(bctx.lambda),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    user.addctx = &bctx;

    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,  // contrast with fish2
                        3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    // this is the Jacobian of the Poisson equation, thus ONLY APPROXIMATE
    //     ... consider using -snes_fd_color or -snes_mf_operator
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Poisson2DJacobianLocal,&user); CHKERRQ(ierr);
    // FIXME set NGS
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",info.mx,info.my); CHKERRQ(ierr);

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

