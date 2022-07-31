static char help[] =
"Solve nonlinear Liouville-Bratu equation in 2D on a structured-grid.  Option prefix lb_.\n"
"Solves\n"
"  - nabla^2 u - lambda e^u = 0\n"
"on the unit square [0,1]x[0,1] subject to zero Dirichlet boundary conditions.\n"
"Critical value occurs about at lambda = 6.808.  Optional exact solution by\n"
"Liouville (1853) for case lambda=1.0.\n\n";

/*
incredible performance from FAS+NGS, full-cycles, also NGS for coarse solve
400 million unknowns in 32 seconds, full 10 digit accuracy
uses all memory of my Thelio massive machine (uses over 90% of 128 Gb; note 20 cores out of 40)
note only 350 flops per degree of freedom!

$ timer mpiexec -n 20 --map-by core --bind-to hwthread ./bratu2D -da_grid_x 6 -da_grid_y 6 -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4 -da_refine 12
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 1
flops = 1.467e+11,  residual calls = 416,  NGS calls = 208
done on 20481 x 20481 grid:   error |u-uexact|_inf = 1.728e-10
real 31.50
*/

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

/* excellent evidence of convergence and optimality in Liouville exact solution case (use opt PETSc build):
NEWTON+MG is fast
$ for LEV in 4 5 6 7 8 9 10; do timer ./bratu2D -da_refine $LEV -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type newtonls -snes_fd_color -pc_type mg; done
FAS+NGS full cycles are much faster
$ for LEV in 4 5 6 7 8 9 10; do timer ./bratu2D -da_refine $LEV -lb_exact -snes_rtol 1.0e-10 -snes_converged_reason -lb_showcounts -snes_type fas -snes_fas_type full -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 2 -fas_levels_snes_max_it 1 -fas_coarse_snes_type ngs -fas_coarse_snes_ngs_sweeps 2 -fas_coarse_snes_max_it 4; done
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
    return PetscLogReal(32.0 * omega);
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, Vec, PoissonCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal **,
                                        PetscReal**, PoissonCtx*);
extern PetscErrorCode NonlinearGS(SNES, Vec, Vec, void*);

int main(int argc,char **argv) {
    DM             da, da_after;
    SNES           snes;
    Vec            u, uexact;
    PoissonCtx     user;
    BratuCtx       bctx;
    DMDALocalInfo  info;
    PetscBool      showcounts = PETSC_FALSE;
    PetscLogDouble lflops, flops;
    PetscReal      errinf;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));
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
    PetscOptionsBegin(PETSC_COMM_WORLD,"lb_","Liouville-Bratu equation solver options","");
    PetscCall(PetscOptionsReal("-lambda","coefficient of e^u (reaction) term",
                            "bratu2D.c",bctx.lambda,&(bctx.lambda),NULL));
    PetscCall(PetscOptionsBool("-exact","use case of Liouville exact solution",
                            "bratu2D.c",bctx.exact,&(bctx.exact),NULL));
    PetscCall(PetscOptionsBool("-showcounts","at finish, print numbers of calls to call-back functions",
                            "bratu2D.c",showcounts,&showcounts,NULL));
    PetscOptionsEnd();
    if (bctx.exact) {
        if (bctx.lambda != 1.0) {
            SETERRQ(PETSC_COMM_SELF,1,"Liouville exact solution only implemented for lambda = 1.0\n");
        }
        user.g_bdry = &g_liouville;
    }
    user.addctx = &bctx;

    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,  // contrast with fish2
                        3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
    PetscCall(DMSetApplicationContext(da,&user));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));  // this must be called BEFORE SetUniformCoordinates
    PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user));
    PetscCall(SNESSetNGS(snes,NonlinearGS,&user));
    // this is the Jacobian of the Poisson equation, thus ONLY APPROXIMATE
    //     ... consider using -snes_fd_color or -snes_mf_operator
    PetscCall(DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Poisson2DJacobianLocal,&user));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(DMGetGlobalVector(da,&u));
    PetscCall(VecSet(u,0.0));  // initialize to zero
    PetscCall(SNESSolve(snes,NULL,u));
    PetscCall(DMRestoreGlobalVector(da,&u));
    PetscCall(DMDestroy(&da));

    if (showcounts) {
        PetscCall(PetscGetFlops(&lflops));
        PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,PetscObjectComm((PetscObject)snes)));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"flops = %.3e,  residual calls = %d,  NGS calls = %d\n",
                           flops,bctx.residualcount,bctx.ngscount));
    }

    PetscCall(SNESGetDM(snes,&da_after));
    PetscCall(DMDAGetLocalInfo(da_after,&info));
    if (bctx.exact) {
        PetscCall(SNESGetSolution(snes,&u));  // SNES owns u; we do not destroy it
        PetscCall(DMGetGlobalVector(da_after,&uexact));
        PetscCall(FormUExact(&info,uexact,&user));
        PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uexact
        PetscCall(DMRestoreGlobalVector(da_after,&uexact));
        PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "done on %d x %d grid:   error |u-uexact|_inf = %.3e\n",
                info.mx,info.my,errinf));
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",info.mx,info.my));
    }

    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
    BratuCtx     *bctx = (BratuCtx*)(user->addctx);
    PetscInt     i, j;
    PetscReal    hx, hy, x, y, **au;
    if (user->g_bdry != &g_liouville) {
        SETERRQ(PETSC_COMM_SELF,1,"exact solution only implemented for g_liouville() boundary conditions\n");
    }
    if (bctx->lambda != 1.0) {
        SETERRQ(PETSC_COMM_SELF,2,"Liouville exact solution only implemented for lambda = 1.0\n");
    }
    hx = 1.0 / (PetscReal)(info->mx - 1);
    hy = 1.0 / (PetscReal)(info->my - 1);
    PetscCall(DMDAVecGetArray(info->da, u, &au));
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = i * hx;
            au[j][i] = user->g_bdry(x,y,0.0,bctx);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da, u, &au));
    return 0;
}

// compute F(u), the residual of the discretized PDE on the given grid
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PoissonCtx *user) {
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
    PetscCall(PetscLogFlops(12.0 * info->xm * info->ym));
    (bctx->residualcount)++;
    return 0;
}

// do nonlinear Gauss-Seidel (processor-block) sweeps on
//     F(u) = b
PetscErrorCode NonlinearGS(SNES snes, Vec u, Vec b, void *ctx) {
    PetscInt       i, j, k, maxits, totalits=0, sweeps, l;
    PetscReal      atol, rtol, stol, hx, hy, darea, hxhy, hyhx, x, y,
                   **au, **ab, bij, uu, phi0, phi, dphidu, s;
    DM             da;
    DMDALocalInfo  info;
    PoissonCtx     *user = (PoissonCtx*)(ctx);
    BratuCtx       *bctx = (BratuCtx*)(user->addctx);
    Vec            uloc;

    PetscCall(SNESNGSGetSweeps(snes,&sweeps));
    PetscCall(SNESNGSGetTolerances(snes,&atol,&rtol,&stol,&maxits));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(DMDAGetLocalInfo(da,&info));

    hx = 1.0 / (PetscReal)(info.mx - 1);
    hy = 1.0 / (PetscReal)(info.my - 1);
    darea = hx * hy;
    hxhy = hx / hy;
    hyhx = hy / hx;

    PetscCall(DMGetLocalVector(da,&uloc));
    for (l=0; l<sweeps; l++) {
        PetscCall(DMGlobalToLocalBegin(da,u,INSERT_VALUES,uloc));
        PetscCall(DMGlobalToLocalEnd(da,u,INSERT_VALUES,uloc));
        PetscCall(DMDAVecGetArray(da,uloc,&au));
        if (b) {
            PetscCall(DMDAVecGetArrayRead(da,b,&ab));
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
        PetscCall(DMDAVecRestoreArray(da,uloc,&au));
        PetscCall(DMLocalToGlobalBegin(da,uloc,INSERT_VALUES,u));
        PetscCall(DMLocalToGlobalEnd(da,uloc,INSERT_VALUES,u));
    }
    PetscCall(DMRestoreLocalVector(da,&uloc));
    if (b) {
        PetscCall(DMDAVecRestoreArrayRead(da,b,&ab));
    }
    PetscCall(PetscLogFlops(21.0 * totalits));
    (bctx->ngscount)++;
    return 0;
}
