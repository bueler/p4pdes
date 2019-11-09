static char help[] =
"Solve nonlinear Liouville-Bratu equation in 2D on a structured-grid.  Option prefix lb_.\n"
"Solves\n"
"  - nabla^2 u - lambda e^u = 0\n"
"on the unit square [0,1]x[0,1] subject to zero Dirichlet boundary conditions.\n"
"Critical value occurs about at lambda = 6.808.  Optional exact solution\n"
"(Liouville 1853) in case lambda=1.0.\n\n";

/* compare:
timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8

timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8 -snes_fd_color

timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -da_refine 8 -snes_mf_operator

timer ./bratu2D -snes_monitor -snes_converged_reason -ksp_converged_reason -pc_type mg -snes_grid_sequence 8

timer ./bratu2D -snes_monitor -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type ngs

timer mpiexec -n 4 ./bratu2D -snes_monitor -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type ngs

timer ./bratu2D -snes_monitor -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type ngs -fas_levels_snes_ngs_sweeps 2

timer ./bratu2D -snes_monitor -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type preonly -fas_coarse_pc_type cholesky

timer ./bratu2D -snes_monitor -snes_converged_reason -lb_showcounts -da_refine 8 -snes_type fas -fas_levels_snes_type ngs -fas_coarse_snes_type newtonls -fas_coarse_ksp_type cg -fas_coarse_pc_type icc -snes_fas_monitor -snes_fas_levels 6
*/

/* excellent evidence of convergence in Liouville exact solution case:
$ for LEV in 3 4 5 6 7 8 9; do ./bratu2D -da_refine $LEV -snes_monitor -snes_fd_color -snes_rtol 1.0e-10 -lb_exact -pc_type mg; done
*/

#include <petsc.h>
#include "../../ch6/poissonfunctions.h"

typedef struct {
    PetscReal lambda;
    PetscBool exact;
    int       residualcount, ngscount;
} BratuCtx;

static PetscReal g_zero(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return 0.0;
}

static PetscReal g_liouville(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    PetscReal r2 = (x + 1.0) * (x + 1.0) + (y + 1.0) * (y + 1.0),
              qq = r2 * r2 + 1.0,
              omega = r2 / (qq * qq);
    return 32.0 * omega;
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, Vec, PoissonCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal **,
                                        PetscReal**, PoissonCtx*);
extern PetscErrorCode NonlinearGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u, uexact;
    PoissonCtx     user;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      showcounts = PETSC_FALSE;
    PetscLogDouble flops;
    PetscReal      errinf;

    PetscInitialize(&argc,&argv,NULL,help);
    user.Lx = 1.0;
    user.Ly = 1.0;
    user.Lz = 1.0;
    user.cx = 1.0;
    user.cy = 1.0;
    user.cz = 1.0;
    user.g_bdry = &g_zero;
    bctx.lambda = 1.0;
    bctx.exact = PETSC_FALSE;
    bctx.residualcount = 0;
    bctx.ngscount = 0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu2D.c",bctx.lambda,&(bctx.lambda),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exact","use case of Liouville exact solution",
                            "bratu2D.c",bctx.exact,&(bctx.exact),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-showcounts","at finish, print numbers of calls to call-back functions",
                            "bratu2D.c",showcounts,&showcounts,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    if (bctx.exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_WORLD,1,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        user.g_bdry = &g_liouville;
    }
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
    ierr = SNESSetNGS(snes,NonlinearGS,&user); CHKERRQ(ierr);
    // this is the Jacobian of the Poisson equation, thus ONLY APPROXIMATE
    //     ... consider using -snes_fd_color or -snes_mf_operator
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Poisson2DJacobianLocal,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);  // initialize to zero
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&u);CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    if (showcounts) {
        ierr = PetscGetFlops(&flops); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"flops = %.3e,  residual calls = %d,  NGS calls = %d\n",
                           flops,bctx.residualcount,bctx.ngscount); CHKERRQ(ierr);
    }

    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    if (bctx.exact) {
        ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);  // SNES owns u; we do not destroy it
        ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
        ierr = FormUExact(&info,uexact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
        ierr = VecDestroy(&uexact); CHKERRQ(ierr);  // no longer needed
        ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                info.mx,info.my,errinf); CHKERRQ(ierr);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",info.mx,info.my); CHKERRQ(ierr);
    }

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
    PetscErrorCode ierr;
    BratuCtx     *bctx = (BratuCtx*)(user->addctx);
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    if (user->g_bdry != &g_liouville) {
        SETERRQ(PETSC_COMM_WORLD,1,"exact solution only implemented for g_liouville() boundary conditions\n");
    }
    if (bctx->lambda != 1.0) {
        SETERRQ(PETSC_COMM_WORLD,2,"Liouville exact solution only implemented for lambda = 1.0\n");
    }
    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;
            au[j][i] = user->g_bdry(x,y,0.0,bctx);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
    return 0;
}

// compute F(u), the residual of the discretized PDE on the given grid
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PoissonCtx *user) {
    PetscErrorCode ierr;
    BratuCtx   *bctx = (BratuCtx*)(user->addctx);
    PetscInt   i, j;
    PetscReal  hx, hy, darea, hxhy, hyhx, x, y;

    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    darea = hx * hy;
    hxhy = hx / hy;
    hyhx = hy / hx;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            if (j==0 || i==0 || i==info->mx-1 || j==info->my-1) {
                x = i * hx;
                FF[j][i] = au[j][i] - user->g_bdry(x,y,0.0,bctx);
            } else {
                FF[j][i] =   hyhx * (2.0 * au[j][i] - au[j][i-1] - au[j][i+1])
                           + hxhy * (2.0 * au[j][i] - au[j-1][i] - au[j+1][i])
                           - darea * bctx->lambda * PetscExpScalar(au[j][i]);
            }
        }
    }
    ierr = PetscLogFlops(12.0 * info->xm * info->ym); CHKERRQ(ierr);
    (bctx->residualcount)++;
    return 0;
}

// do nonlinear Gauss-Seidel (processor-block) sweeps on
//     F(u) = b
PetscErrorCode NonlinearGS(SNES snes, Vec u, Vec b, void *ctx) {
    PetscErrorCode ierr;
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, darea, hxhy, hyhx, x, y,
                   **au, **ab, bij, uu, phi0, phi, dphidu, s;
    DM             da;
    DMDALocalInfo  info;
    PoissonCtx     *user = (PoissonCtx*)(ctx);
    BratuCtx       *bctx = (BratuCtx*)(user->addctx);
    Vec            uloc;

    ierr = SNESNGSGetSweeps(snes,&sweeps);CHKERRQ(ierr);
    ierr = SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits);CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
    darea = hx * hy;
    hxhy = hx / hy;
    hyhx = hy / hx;

    ierr = DMGetLocalVector(da,&uloc);CHKERRQ(ierr);
    for (l=0; l<sweeps; l++) {
        ierr = DMGlobalToLocalBegin(da,u,INSERT_VALUES,uloc);CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(da,u,INSERT_VALUES,uloc);CHKERRQ(ierr);
        ierr = DMDAVecGetArray(da,uloc,&au);CHKERRQ(ierr);
        if (b) {
            ierr = DMDAVecGetArrayRead(da,b,&ab); CHKERRQ(ierr);
        }
        for (j = info.ys; j < info.ys + info.ym; j++) {
            y = j * hy;
            for (i = info.xs; i < info.xs + info.xm; i++) {
                if (j==0 || i==0 || i==info.mx-1 || j==info.my-1) {
                    x = i * hx;
                    au[j][i] = user->g_bdry(x,y,0.0,bctx);
                } else {
                    if (b)
                        bij = ab[j][i];
                    else
                        bij = 0.0;
                    // do pointwise Newton iterations on scalar function
                    //   phi(u) =   hyhx * (2 u - au[j][i-1] - au[j][i+1])
                    //            + hxhy * (2 u - au[j-1][i] - au[j+1][i])
                    //            - darea * lambda * e^u - bij
                    uu = au[j][i];
                    phi0 = 0.0;
                    for (k = 0; k < maxits; k++) {
                        phi =   hyhx * (2.0 * uu - au[j][i-1] - au[j][i+1])
                              + hxhy * (2.0 * uu - au[j-1][i] - au[j+1][i])
                              - darea * bctx->lambda * PetscExpScalar(uu) - bij;
                        if (k == 0)
                             phi0 = phi;
                        dphidu = 2.0 * (hyhx + hxhy)
                                 - darea * bctx->lambda * PetscExpScalar(uu);
                        s = - phi / dphidu;     // Newton step
                        uu += s;
                        totalits++;
                        if (   atol > PetscAbsReal(phi)
                            || rtol*PetscAbsReal(phi0) > PetscAbsReal(phi)
                            || stol*PetscAbsReal(uu) > PetscAbsReal(s)    ) {
                            break;
                        }
                    }
                    au[j][i] = uu;
                }
            }
        }
        ierr = DMDAVecRestoreArray(da,uloc,&au);CHKERRQ(ierr);
        ierr = DMLocalToGlobalBegin(da,uloc,INSERT_VALUES,u);CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd(da,uloc,INSERT_VALUES,u);CHKERRQ(ierr);
    }
    ierr = DMRestoreLocalVector(da,&uloc);CHKERRQ(ierr);
    if (b) {
        ierr = DMDAVecRestoreArrayRead(da,b,&ab);CHKERRQ(ierr);
    }
    ierr = PetscLogFlops(21.0 * totalits); CHKERRQ(ierr);
    (bctx->ngscount)++;
    return 0;
}

