static char help[] =
"Structured-grid minimal surface equation (MSE) in 2D.  Option prefix mse_.\n"
"Equation is\n"
"  - div ( (1 + |grad u|^2)^q grad u ) = 0\n"
"on the unit square [0,1]x[0,1] subject to Dirichlet boundary\n"
"conditions u = g(x,y).  Power q defaults to -1/2 (i.e. MSE) but is adjustable.\n"
"Catenoid and tent boundary conditions are implemented; catenoid is an exact\n"
"solution.  We re-use the Jacobian from the Poisson equation, but it is suitable\n"
"only for low-amplitude g, or as preconditioning material in -snes_mf_operator.\n"
"Options -snes_fd_color and -snes_grid_sequence K are recommended.\n"
"This code is multigrid (GMG) capable.\n\n";

/* 
-snes_grid_sequence is effective when nonlinearity is strong (default H=1 is
already strong!) and for getting into domain of convergence in catenoid case

this would seem to be FAS with NGS smoothing, but with "Use finite difference secant approximation with coloring with h = 1e-08" for the NGS, which is also used on coarse grid:
./minimal -da_refine 4 -snes_monitor -snes_type fas -fas_coarse_snes_type ngs -fas_levels_snes_type ngs

also:
  ./minimal -da_refine 4 -snes_monitor -snes_type fas -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -fas_coarse_snes_type newtonls -fas_levels_snes_type nrichardson
  ./minimal -da_refine 4 -snes_monitor -snes_type fas -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -fas_coarse_snes_type newtonls -fas_levels_snes_type ngs

with tent?, can't seem to speed up in trying to reduce ksp iterations on finer levels:
    * try -pc_mg_type full or -pc_mg_cycle_type w   [neither very effective]
    * change the amount of smoothing: -mg_levels_ksp_max_it 10   [slower but counts down]
    * change the smoother: -mg_levels_pc_type gamg  (!)    [slower but counts a lot down]
*/

#include <petsc.h>
#include "poissonfunctions.h"
#include "../quadrature.h"

typedef struct {
    double    q,          // the exponent in the diffusivity;
                          // =-1/2 for minimal surface eqn; =0 for Laplace eqn
              tent_H,     // height of tent door along y=0 boundary
              catenoid_c; // parameter in catenoid formula
} MinimalCtx;

// Dirichlet boundary conditions
static double g_bdry_tent(double x, double y, double z, void *ctx) {
    PoissonCtx *user = (PoissonCtx*)ctx;
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    if (x < 1.0e-8) {
        return 2.0 * mctx->tent_H * (y < 0.5 ? y : 1.0 - y);
    } else
        return 0;
}

static double g_bdry_catenoid(double x, double y, double z, void *ctx) {
    PoissonCtx   *user = (PoissonCtx*)ctx;
    MinimalCtx   *mctx = (MinimalCtx*)(user->addctx);
    const double c = mctx->catenoid_c;
    return c * cosh(x/c) * sin(acos( (y/c) / cosh(x/c) ));
}

// the coefficient (diffusivity) of minimal surface equation, as a function
//   of  w = |grad u|^2
static double DD(double w, double q) {
    return pow(1.0 + w,q);
}

extern PetscErrorCode FormExactFromG(DMDALocalInfo*, Vec, PoissonCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double**,
                                        double **FF, PoissonCtx*);
extern PetscErrorCode MSEMonitor(SNES, int, double, void*);

typedef enum {TENT, CATENOID} ProblemType;
static const char* ProblemTypes[] = {"tent","catenoid",
                                     "ProblemType", "", NULL};

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u_initial, u;
    PoissonCtx     user;
    MinimalCtx     mctx;
    PetscBool      monitor = PETSC_FALSE,
                   exact_init = PETSC_FALSE;
    DMDALocalInfo  info;
    ProblemType    problem = CATENOID;

    // defaults:
    mctx.q = -0.5;
    mctx.tent_H = 1.0;
    mctx.catenoid_c = 1.1;  // case shown in Figure in book
    user.cx = 1.0;
    user.cy = 1.0;
    user.cz = 1.0;

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"mse_",
                             "minimal surface equation solver options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-catenoid_c",
                            "parameter for problem catenoid; c >= 1 required",
                            "minimal.c",mctx.catenoid_c,&(mctx.catenoid_c),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-exact_init",
                            "initial Newton iterate = continuum exact solution; only for catenoid",
                            "minimal.c",exact_init,&(exact_init),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-monitor",
                            "print surface area and diffusivity bounds at each SNES iteration",
                            "minimal.c",monitor,&(monitor),NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-q",
                            "power of (1+|grad u|^2) in diffusivity",
                            "minimal.c",mctx.q,&(mctx.q),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem",
                            "problem type determines boundary conditions",
                            "minimal.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,
                            NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tent_H",
                            "'door' height for problem tent",
                            "minimal.c",mctx.tent_H,&(mctx.tent_H),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user.addctx = &mctx;   // attach MSE-specific parameters
    switch (problem) {
        case TENT:
            if (exact_init) {
                SETERRQ(PETSC_COMM_WORLD,2,
                    "initialization with exact solution only possible for -mse_problem catenoid\n");
            }
            user.g_bdry = &g_bdry_tent;
            break;
        case CATENOID:
            if (mctx.catenoid_c < 1.0) {
                SETERRQ(PETSC_COMM_WORLD,3,
                    "catenoid exact solution only valid if c >= 1\n");
            }
            if ((exact_init) && (mctx.q != -0.5)) {
                SETERRQ(PETSC_COMM_WORLD,4,
                    "initialization with catenoid exact solution only possible if q=-0.5\n");
            }
            user.g_bdry = &g_bdry_catenoid;
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,5,"unknown problem type\n");
    }

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
    // this is the Jacobian of the Poisson equation, thus ONLY APPROXIMATE;
    //     generally use -snes_fd_color or -snes_mf_operator
    ierr = DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Poisson2DJacobianLocal,&user); CHKERRQ(ierr);
    if (monitor) {
        ierr = SNESMonitorSet(snes,MSEMonitor,&user,NULL); CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&u_initial); CHKERRQ(ierr);
    if ((problem == CATENOID) && (mctx.q == -0.5) && (exact_init)) {
        ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
        ierr = FormExactFromG(&info,u_initial,&user); CHKERRQ(ierr);
    } else {
        // initial iterate has u=g on boundary and u=0 in interior
        ierr = InitialState(da, ZEROS, PETSC_TRUE, u_initial, &user); CHKERRQ(ierr);
    }

//STARTSNESSOLVE
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid and problem %s",
                       info.mx,info.my,ProblemTypes[problem]); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
//ENDSNESSOLVE

    // evaluate numerical error in exact solution case
    if ((problem == CATENOID) && (mctx.q == -0.5)) {
        Vec    u_exact;
        double errnorm;
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormExactFromG(&info,u_exact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
        ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
        ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                           ":  error |u-uexact|_inf = %.5e\n",errnorm); CHKERRQ(ierr);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD," ...\n"); CHKERRQ(ierr);
    }

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormExactFromG(DMDALocalInfo *info, Vec uexact,
                         PoissonCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  xymin[2], xymax[2], hx, hy, x, y, **auexact;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da,uexact,&auexact); CHKERRQ(ierr);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            auexact[j][i] = user->g_bdry(x,y,0.0,user);
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
    double     xymin[2], xymax[2], hx, hy, hxhy, hyhx, x, y,
               ue, uw, un, us, une, use, unw, usw,
               dux, duy, De, Dw, Dn, Ds;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
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
                De = DD(dux * dux + duy * duy, mctx->q);
                // ...                   at west point  (i-1/2,j):
                dux = (au[j][i] - uw) / hx;
                duy = (unw + un - usw - us) / (4.0 * hy);
                Dw = DD(dux * dux + duy * duy, mctx->q);
                // ...                  at north point  (i,j+1/2):
                dux = (ue + une - uw - unw) / (4.0 * hx);
                duy = (un - au[j][i]) / hy;
                Dn = DD(dux * dux + duy * duy, mctx->q);
                // ...                  at south point  (i,j-1/2):
                dux = (ue + use - uw - usw) / (4.0 * hx);
                duy = (au[j][i] - us) / hy;
                Ds = DD(dux * dux + duy * duy, mctx->q);
                // evaluate residual
                FF[j][i] = - hyhx * (De * (ue - au[j][i]) - Dw * (au[j][i] - uw))
                           - hxhy * (Dn * (un - au[j][i]) - Ds * (au[j][i] - us));
            }
        }
    }
    return 0;
}

// compute surface area and bounds on diffusivity using Q^1 elements and
// tensor product gaussian quadrature
PetscErrorCode MSEMonitor(SNES snes, int its, double norm, void *user) {
    PetscErrorCode ierr;
    PoissonCtx     *pctx = (PoissonCtx*)(user);
    MinimalCtx     *mctx = (MinimalCtx*)(pctx->addctx);
    DM             da;
    Vec            u, uloc;
    DMDALocalInfo  info;
    const int      ndegree = 2;
    const Quad1D   q = gausslegendre[ndegree-1];   // from ../quadrature.h
    double         xymin[2], xymax[2], hx, hy, **au, x_i, y_j, x, y,
                   ux, uy, W, D,
                   Dminloc = PETSC_INFINITY, Dmaxloc = 0.0, Dmin, Dmax,
                   arealoc = 0.0, area;
    int            i, j, r, s;
    MPI_Comm       comm;
    ierr = SNESGetDM(snes, &da); CHKERRQ(ierr);
    ierr = SNESGetSolution(snes, &u); CHKERRQ(ierr);
    ierr = DMGetLocalVector(da, &uloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, u, INSERT_VALUES, uloc); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, u, INSERT_VALUES, uloc); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = DMDAGetBoundingBox(info.da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info.mx - 1);
    hy = (xymax[1] - xymin[1]) / (info.my - 1);
    ierr = DMDAVecGetArrayRead(da,uloc,&au); CHKERRQ(ierr);
//STARTMONITORLOOP
    // loop over rectangular cells in grid
    for (j = info.ys; j < info.ys + info.ym; j++) {
        if (j == 0)
            continue;
        y_j = j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if (i == 0)
                continue;
            x_i = i * hx;
            // loop over quadrature points in rectangle w NE corner (x_i,y_j)
            for (r = 0; r < q.n; r++) {
                x = x_i - hx + hx * 0.5 * (q.xi[r] + 1);
                for (s = 0; s < q.n; s++) {
                    y = y_j - hy + hy * 0.5 * (q.xi[s] + 1);
                    // gradient of u(x,y) at quadrature points
                    ux =   (au[j][i] - au[j][i-1])     * (y - (y_j - hy))
                         + (au[j-1][i] - au[j-1][i-1]) * (y_j - y);
                    ux /= hx * hy;
                    uy =   (au[j][i] - au[j-1][i])     * (x - (x_i - hx))
                         + (au[j][i-1] - au[j-1][i-1]) * (x_i - x);
                    uy /= hx * hy;
                    W = ux * ux + uy * uy;
                    // min and max of diffusivity at quadrature points
                    D = DD(W,mctx->q);
                    Dminloc = PetscMin(Dminloc,D);
                    Dmaxloc = PetscMax(Dmaxloc,D);
                    // apply quadrature in surface area formula
                    arealoc += q.w[r] * q.w[s] * PetscSqrtReal(1.0 + W);
                }
            }
        }
    }
//ENDMONITORLOOP
    ierr = DMDAVecRestoreArrayRead(da,uloc,&au); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &uloc); CHKERRQ(ierr);

    // do global reductions and report
    ierr = PetscObjectGetComm((PetscObject)da,&comm); CHKERRQ(ierr);
    arealoc *= hx * hy / 4.0;  // from change of variables formula
    ierr = MPI_Allreduce(&arealoc,&area,1,MPI_DOUBLE,MPI_SUM,comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&Dminloc,&Dmin,1,MPI_DOUBLE,MPI_MIN,comm); CHKERRQ(ierr);
    ierr = MPI_Allreduce(&Dmaxloc,&Dmax,1,MPI_DOUBLE,MPI_MAX,comm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"area = %.8f; %.4f <= D(|grad u|^2) <= %.4f\n",
               area,Dmin,Dmax); CHKERRQ(ierr);
    return 0;
}

