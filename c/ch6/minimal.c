static char help[] =
"Structured-grid minimal surface equation in 2D.  Option prefix mse_.\n"
"Solves\n"
"            /         nabla u         \\ \n"
"  - nabla . | ----------------------- | = f(x,y)\n"
"            \\  sqrt(1 + |nabla u|^2)  / \n"
"on the unit square [0,1]x[0,1], subject to Dirichlet boundary conditions\n"
"u = g(x,y).  Main example has \"tent\" boundary condition.  Nonzero RHS is\n"
"only used in an optional manufactured solution.  Re-uses of Jacobian from\n"
"Poisson equation (see fish2.c) as preconditioner; this is suitable only for\n"
"low-amplitude data (g and f).  Multigrid-capable.\n\n";

/* 
snes_fd_color is ten times faster than snes_mf_operator?  (and multigrid gives some speedup over ilu):
    $ timer ./minimal -snes_mf_operator -snes_converged_reason -da_refine 6
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 27
    done on 129 x 129 grid ...
    real 90.47
    $ timer ./minimal -snes_fd_color -snes_converged_reason -da_refine 6
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 28
    done on 129 x 129 grid ...
    real 8.09
    $ timer ./minimal -snes_mf_operator -snes_converged_reason -da_refine 6 -ksp_type gmres -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 27
    done on 129 x 129 grid ...
    real 49.46
    $ timer ./minimal -snes_fd_color -snes_converged_reason -da_refine 6 -ksp_type gmres -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 28
    done on 129 x 129 grid ...
    real 6.06

parallel multigrid choices for linear and nonlinear problems; note GMRES vs CG; note also similar performance for fish2 and for minimal -mse_laplace:
    $ timer mpiexec -n 2 ./minimal -snes_fd_color -snes_converged_reason -da_refine 7 -ksp_type gmres -pc_type mg -snes_max_it 500
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 65
    done on 257 x 257 grid ...
    real 59.47
    $ timer mpiexec -n 2 ./minimal -mse_laplace -snes_converged_reason -da_refine 7 -ksp_type cg -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
    done on 257 x 257 grid ...
    real 0.97
    $ timer mpiexec -n 2 ./fish2 -snes_converged_reason -da_refine 7 -ksp_type cg -pc_type mg
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 2
    on 257 x 257 grid:  error |u-uexact|_inf = 7.68279e-07
    real 0.72

evidence of parallel:
    timer mpiexec -n N ./minimal -snes_fd_color -snes_converged_reason -da_refine 6 -pc_type mg -snes_max_it 200
    Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 34
gives 4.0 sec on N=1 and 2.1 sec on N=2

FIXME change code so -snes_grid_sequence will work, and demonstrate;
key idea:   -snes_grid_sequence X   REPLACES!   -da_refine X
*/

#include <petsc.h>
#include "poissonfunctions.h"
#include "../quadrature.h"

typedef struct {
    double    H;       // height of tent along y=0 boundary
    PetscBool laplace; // solve Laplace equation instead of minimal surface
} MinimalCtx;

// Dirichlet boundary condition; only use along y=0 boundary
double g_bdry_tent(double x, double y, double z, void *ctx) {
    PoissonCtx *user = (PoissonCtx*)ctx;
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    if (PetscAbs(y) < 1.0e-8) {
        return 2.0 * mctx->H * (x < 0.5 ? x : 1.0 - x);
    } else
        return 0;
}

// the coefficient (diffusivity) of minimal surface equation, as a function
//   of  s = |grad u|^2
double DD(double s) {
    return pow(1.0 + s,-0.5);
}

// derivative is only used in manufacturing a solution
double dDD(double s) {
    return -0.5 * pow(1.0 + s,-1.5);
}

double zero(double x, double y, double z, void *ctx) {
    return 0.0;
}

// manufactured only: the exact solution u(x,y)
double u_manu(double x, double y, double z, void *ctx) {
    return (x - x*x) * (y*y - y);
}

// manufactured only: the right-hand-side f(x,y) = - div (DD(s) grad u)
double f_manu_laplace(double x, double y, double z, void *ctx) {
    double uxx, uyy;
    uxx  = - 2.0 * (y*y - y);
    uyy  = (x - x*x) * 2.0;
    return - uxx - uyy;
}

double f_manu(double x, double y, double z, void *ctx) {
    double uxx, uyy, ux, uy, uxy, s, sx, sy;
    uxx  = - 2.0 * (y*y - y);
    uyy  = (x - x*x) * 2.0;
    ux   = (1.0 - 2.0 * x) * (y*y - y);
    uy   = (x - x*x) * (2.0 * y - 1.0);
    uxy  = (1.0 - 2.0 * x) * (2.0 * y - 1.0);
    s    = ux * ux + uy * uy;  // s = |grad u|^2
    sx   = 2.0 * ux * uxx + 2.0 * uy * uxy;
    sy   = 2.0 * ux * uxy + 2.0 * uy * uyy;
    return - dDD(s) * (sx * ux + sy * uy) - DD(s) * (uxx + uyy);
}

PetscErrorCode Spacings(DMDALocalInfo *info, double *hx, double *hy) {
    PetscErrorCode ierr;
    double  xymin[2], xymax[2];
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    *hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    *hy = (xymax[1] - xymin[1]) / (info->my - 1);
    return 0;
}

PetscErrorCode FormExactManufactured(DMDALocalInfo *info, Vec uexact,
                                     PoissonCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  hx, hy, x, y, **auexact;
    ierr = Spacings(info,&hx,&hy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(info->da,uexact,&auexact); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            auexact[j][i] = u_manu(x,y,0.0,user);
        }
    }
    ierr = DMDAVecRestoreArray(info->da,uexact,&auexact); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, PoissonCtx *user) {
    PetscErrorCode ierr;
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    int        i, j;
    double     hx, hy, hxhy, hyhx, x, y,
               ue, uw, un, us, une, use, unw, usw,
               dux, duy, De, Dw, Dn, Ds;
    ierr = Spacings(info,&hx,&hy); CHKERRQ(ierr);
    hxhy = hx / hy;
    hyhx = hy / hx;
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            if (j==0 || i==0 || i==info->mx-1 || j==info->my-1) {
                FF[j][i] = au[j][i] - user->g_bdry(x,y,0.0,user);
            } else {
                // assign neighbor values with either boundary condition or
                //     current u at that point (==> symmetric matrix)
                ue  = (i+1 == info->mx-1) ? user->g_bdry(x+hx,y,0.0,user)
                                          : au[j][i+1];
                uw  = (i-1 == 0)          ? user->g_bdry(x-hx,y,0.0,user)
                                          : au[j][i-1];
                un  = (j+1 == info->my-1) ? user->g_bdry(x,y+hy,0.0,user)
                                          : au[j+1][i];
                us  = (j-1 == 0)          ? user->g_bdry(x,y-hy,0.0,user)
                                          : au[j-1][i];
                if (mctx->laplace) {
                    De = 1.0;  Dw = 1.0;
                    Dn = 1.0;  Ds = 1.0;
                } else {
                    if (i+1 == info->mx-1 || j+1 == info->my-1) {
                        une = user->g_bdry(x+hx,y+hy,0.0,user);
                    } else {
                        une = au[j+1][i+1];
                    }
                    if (i-1 == 0 || j+1 == info->my-1) {
                        unw = user->g_bdry(x-hx,y+hy,0.0,user);
                    } else {
                        unw = au[j+1][i-1];
                    }
                    if (i+1 == info->mx-1 || j-1 == 0) {
                        use = user->g_bdry(x+hx,y-hy,0.0,user);
                    } else {
                        use = au[j-1][i+1];
                    }
                    if (i-1 == 0 || j-1 == 0) {
                        usw = user->g_bdry(x-hx,y-hy,0.0,user);
                    } else {
                        usw = au[j-1][i-1];
                    }
                    // gradient  (dux,duy)   at east point  (i+1/2,j):
                    dux = (ue - au[j][i]) / hx;
                    duy = (un + une - us - use) / (4.0 * hy);
                    De = DD(dux * dux + duy * duy);
                    // ...                   at west point  (i-1/2,j):
                    dux = (au[j][i] - uw) / hx;
                    duy = (unw + un - usw - us) / (4.0 * hy);
                    Dw = DD(dux * dux + duy * duy);
                    // ...                  at north point  (i,j+1/2):
                    dux = (ue + une - uw - unw) / (4.0 * hx);
                    duy = (un - au[j][i]) / hy;
                    Dn = DD(dux * dux + duy * duy);
                    // ...                  at south point  (i,j-1/2):
                    dux = (ue + use - uw - usw) / (4.0 * hx);
                    duy = (au[j][i] - us) / hy;
                    Ds = DD(dux * dux + duy * duy);
                }
                // evaluate residual
                FF[j][i] = - hyhx * (De * (ue - au[j][i]) - Dw * (au[j][i] - uw))
                           - hxhy * (Dn * (un - au[j][i]) - Ds * (au[j][i] - us))
                           - hx * hy * user->f_rhs(x,y,0.0,user);
            }
        }
    }
    return 0;
}

// compute surface area using tensor product gaussian quadrature
PetscErrorCode AreaMonitor(SNES snes, int its, double norm, void *ctx) {
    PetscErrorCode ierr;
    DM             da;
    Vec            u, uloc;
    DMDALocalInfo  info;
    const int      ndegree = 2;
    const Quad1D   q = gausslegendre[ndegree-1];
    double         hx, hy, **au, x_i, y_j, x, y, ux, uy, arealoc, area;
    int            i, j, r, s;
    MPI_Comm       comm;
    ierr = SNESGetDM(snes, &da); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u); CHKERRQ(ierr);
    ierr = DMGetLocalVector(da, &uloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, u, INSERT_VALUES, uloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, u, INSERT_VALUES, uloc); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = Spacings(&info,&hx,&hy); CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da,uloc,&au); CHKERRQ(ierr);
    arealoc = 0.0;
    // loop over rectangles in grid
    for (j = info.ys; j < info.ys + info.ym; j++) {
        if (j == 0)
            continue;
        y_j = j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            x_i = i * hx;
            if (i == 0)
                continue;
            // loop over quadrature points in rectangle w corner (x_i,y_j)
            for (r = 0; r < q.n; r++) {
                x = x_i + 0.5 * hx * q.xi[r];
                for (s = 0; s < q.n; s++) {
                    y = y_j + 0.5 * hy * q.xi[s];
                    // slopes of u(x,y) at quadrature point
                    ux =   (au[j][i] - au[j][i-1])     * (y - (y_j - hy))
                         + (au[j-1][i] - au[j-1][i-1]) * (y_j - y);
                    ux /= hx * hy;
                    uy =   (au[j][i] - au[j-1][i])     * (x - (x_i - hx))
                         + (au[j][i-1] - au[j-1][i-1]) * (x_i - x);
                    uy /= hx * hy;
                    // use surface area formula
                    arealoc += q.w[r] * q.w[s]
                               * PetscSqrtReal(1.0 + ux * ux + uy * uy);
                }
            }
        }
    }
    ierr = DMDAVecRestoreArrayRead(da,uloc,&au); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &uloc); CHKERRQ(ierr);
    arealoc *= hx * hy / 4.0;  // from change of variables formula
    ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&arealoc,&area,1,MPI_DOUBLE,MPI_SUM,comm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"    area = %.14f\n",area); CHKERRQ(ierr);
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u;
    PoissonCtx     user;
    MinimalCtx     mctx;
    PetscBool      monitor_area = PETSC_FALSE,
                   manu = PETSC_FALSE;    // solve with manufactured f(x,y) and g(x,y) = 0
    DMDALocalInfo  info;

    PetscInitialize(&argc,&argv,NULL,help);
    mctx.H = 1.0;
    mctx.laplace = PETSC_FALSE;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"mse_","minimal surface equation solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-H","tent height",
                            "minimal.c",mctx.H,&(mctx.H),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-laplace","solve Laplace equation (linear) instead of minimal surface",
                            "minimal.c",mctx.laplace,&(mctx.laplace),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-manu","solve nonlinear problem with manufactured solution",
                            "minimal.c",manu,&(manu),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor_area","compute and print surface area at each SNES iteration",
                            "minimal.c",monitor_area,&(monitor_area),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user.addctx = &mctx;
    if (manu) {
        user.g_bdry = &zero;
        if (mctx.laplace) {
            user.f_rhs = &f_manu_laplace;
        } else {
            user.f_rhs = &f_manu;
        }
    } else {
        user.g_bdry = &g_bdry_tent;
        user.f_rhs = &zero;
    }

    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,  // contrast with fish2
                        3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u"); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
               (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    // this is the Jacobian of the Poisson equation, thus only approximate
    //     (consider using -snes_mf_operator)
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
    if (monitor_area) {
        ierr = SNESMonitorSet(snes,AreaMonitor,NULL,NULL); CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    if (manu) {
        Vec    uexact;
        double errnorm;
        ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
        ierr = FormExactManufactured(&info,uexact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
        ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"error |u-uexact|_inf = %g\n",errnorm); CHKERRQ(ierr);
        VecDestroy(&uexact);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid ...\n",info.mx,info.my); CHKERRQ(ierr);

    VecDestroy(&u);  SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

