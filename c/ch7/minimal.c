static char help[] =
"Solve the minimal surface equation in 2D.  Option prefix ms_.\n"
"Equation is\n"
"  - div ( (1 + |grad u|^2)^q grad u ) = 0\n"
"on the unit square S=(0,1)^2 subject to Dirichlet boundary\n"
"conditions u = g(x,y).  Power q defaults to -1/2 but can be set (by -ms_q).\n"
"Catenoid and tent boundary conditions are implemented; catenoid is an exact\n"
"solution.  The discretization is structured-grid (DMDA) finite differences.\n"
"We re-use the Jacobian from the Poisson equation, but it is suitable only\n"
"for low-amplitude g, or as preconditioning material in -snes_mf_operator.\n"
"Options -snes_fd_color and -snes_grid_sequence are recommended.\n"
"This code is multigrid (GMG) capable.\n\n";

#include <petsc.h>
#include "../ch6/poissonfunctions.h"
#include "../interlude/quadrature.h"

typedef struct {
    PetscReal q,          // the exponent in the diffusivity;
                          //   =-1/2 for minimal surface eqn; =0 for Laplace eqn
              tent_H,     // height of tent door along y=0 boundary
              catenoid_c; // parameter in catenoid formula
    PetscInt  quaddegree; // quadrature degree used in -mse_monitor
} MinimalCtx;

// Dirichlet boundary conditions
static PetscReal g_bdry_tent(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    PoissonCtx *user = (PoissonCtx*)ctx;
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    if (x < 1.0e-8) {
        return 2.0 * mctx->tent_H * (y < 0.5 ? y : 1.0 - y);
    } else
        return 0;
}

static PetscReal g_bdry_catenoid(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    PoissonCtx      *user = (PoissonCtx*)ctx;
    MinimalCtx      *mctx = (MinimalCtx*)(user->addctx);
    const PetscReal c = mctx->catenoid_c;
    return c * PetscCoshReal(x/c)
             * PetscSinReal(PetscAcosReal( (y/c) / PetscCoshReal(x/c) ));
}

// the coefficient (diffusivity) of minimal surface equation, as a function
//   of  w = |grad u|^2
static PetscReal DD(PetscReal w, PetscReal q) {
    return pow(1.0 + w,q);
}

typedef enum {TENT, CATENOID} ProblemType;
static const char* ProblemTypes[] = {"tent","catenoid",
                                     "ProblemType", "", NULL};

extern PetscErrorCode FormExactFromG(DMDALocalInfo*, Vec, PoissonCtx*);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal**,
                                        PetscReal **FF, PoissonCtx*);
extern PetscErrorCode MSEMonitor(SNES, int, PetscReal, void*);

int main(int argc, char **argv) {
    DM             da;
    SNES           snes;
    Vec            u_initial, u;
    PoissonCtx     user;
    MinimalCtx     mctx;
    PetscBool      monitor = PETSC_FALSE,
                   exact_init = PETSC_FALSE;
    DMDALocalInfo  info;
    ProblemType    problem = CATENOID;

    PetscCall(PetscInitialize(&argc,&argv,NULL,help));

    // defaults and options
    mctx.q = -0.5;
    mctx.tent_H = 1.0;
    mctx.catenoid_c = 1.1;  // case shown in Figure in book
    mctx.quaddegree = 3;
    user.cx = 1.0;
    user.cy = 1.0;
    user.cz = 1.0;
    PetscOptionsBegin(PETSC_COMM_WORLD,"ms_",
                      "minimal surface equation solver options","");
    PetscCall(PetscOptionsReal("-catenoid_c",
                            "parameter for problem catenoid; c >= 1 required",
                            "minimal.c",mctx.catenoid_c,&(mctx.catenoid_c),NULL));
    PetscCall(PetscOptionsBool("-exact_init",
                            "initial Newton iterate = continuum exact solution; only for catenoid",
                            "minimal.c",exact_init,&(exact_init),NULL));
    PetscCall(PetscOptionsBool("-monitor",
                            "print surface area and diffusivity bounds at each SNES iteration",
                            "minimal.c",monitor,&(monitor),NULL));
    PetscCall(PetscOptionsReal("-q",
                            "power of (1+|grad u|^2) in diffusivity",
                            "minimal.c",mctx.q,&(mctx.q),NULL));
    PetscCall(PetscOptionsInt("-quaddegree",
                            "quadrature degree (=1,2,3) used in -mse_monitor",
                            "minimal.c",mctx.quaddegree,&(mctx.quaddegree),NULL));
    PetscCall(PetscOptionsEnum("-problem",
                            "problem type determines boundary conditions",
                            "minimal.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,
                            NULL));
    PetscCall(PetscOptionsReal("-tent_H",
                            "'door' height for problem tent",
                            "minimal.c",mctx.tent_H,&(mctx.tent_H),NULL));
    PetscOptionsEnd();

    user.addctx = &mctx;   // attach MSE-specific parameters
    switch (problem) {
        case TENT:
            if (exact_init) {
                SETERRQ(PETSC_COMM_SELF,2,
                    "initialization with exact solution only possible for -mse_problem catenoid\n");
            }
            user.g_bdry = &g_bdry_tent;
            break;
        case CATENOID:
            if (mctx.catenoid_c < 1.0) {
                SETERRQ(PETSC_COMM_SELF,3,
                    "catenoid exact solution only valid if c >= 1\n");
            }
            if ((exact_init) && (mctx.q != -0.5)) {
                SETERRQ(PETSC_COMM_SELF,4,
                    "initialization with catenoid exact solution only possible if q=-0.5\n");
            }
            user.g_bdry = &g_bdry_catenoid;
            break;
        default:
            SETERRQ(PETSC_COMM_SELF,5,"unknown problem type\n");
    }

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
    // this is the Jacobian of the Poisson equation, thus ONLY APPROXIMATE;
    //     generally use -snes_fd_color or -snes_mf_operator
    PetscCall(DMDASNESSetJacobianLocal(da,
               (DMDASNESJacobian)Poisson2DJacobianLocal,&user));
    if (monitor) {
        PetscCall(SNESMonitorSet(snes,MSEMonitor,&user,NULL));
    }
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(DMGetGlobalVector(da,&u_initial));
    if ((problem == CATENOID) && (mctx.q == -0.5) && (exact_init)) {
        PetscCall(DMDAGetLocalInfo(da,&info));
        PetscCall(FormExactFromG(&info,u_initial,&user));
    } else {
        // initial iterate has u=g on boundary and u=0 in interior
        PetscCall(InitialState(da, ZEROS, PETSC_TRUE, u_initial, &user));
    }

//STARTSNESSOLVE
    PetscCall(SNESSolve(snes,NULL,u_initial));
    PetscCall(DMRestoreGlobalVector(da,&u_initial));
    PetscCall(DMDestroy(&da));
    PetscCall(SNESGetDM(snes,&da));
    PetscCall(SNESGetSolution(snes,&u));
//ENDSNESSOLVE

    // evaluate numerical error in exact solution case
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"done on %d x %d grid and problem %s",
                       info.mx,info.my,ProblemTypes[problem]));
    if ((problem == CATENOID) && (mctx.q == -0.5)) {
        Vec    u_exact;
        PetscReal errnorm;
        PetscCall(DMCreateGlobalVector(da,&u_exact));
        PetscCall(FormExactFromG(&info,u_exact,&user));
        PetscCall(VecAXPY(u,-1.0,u_exact));    // u <- u + (-1.0) uexact
        PetscCall(VecDestroy(&u_exact));
        PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                           ":  error |u-uexact|_inf = %.5e\n",errnorm));
    } else {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD," ...\n"));
    }

    PetscCall(SNESDestroy(&snes));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormExactFromG(DMDALocalInfo *info, Vec uexact,
                              PoissonCtx *user) {
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, x, y, **auexact;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    PetscCall(DMDAVecGetArray(info->da,uexact,&auexact));
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = i * hx;
            auexact[j][i] = user->g_bdry(x,y,0.0,user);
        }
    }
    PetscCall(DMDAVecRestoreArray(info->da,uexact,&auexact));
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **FF, PoissonCtx *user) {
    MinimalCtx *mctx = (MinimalCtx*)(user->addctx);
    PetscInt   i, j;
    PetscReal  xymin[2], xymax[2], hx, hy, hxhy, hyhx, x, y,
               ue, uw, un, us, une, use, unw, usw,
               dux, duy, De, Dw, Dn, Ds;
    PetscCall(DMGetBoundingBox(info->da,xymin,xymax));
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

// compute surface area and bounds on diffusivity using Q_1 elements and
// tensor product gaussian quadrature
PetscErrorCode MSEMonitor(SNES snes, PetscInt its, PetscReal norm, void *user) {
    PoissonCtx     *pctx = (PoissonCtx*)(user);
    MinimalCtx     *mctx = (MinimalCtx*)(pctx->addctx);
    DM             da;
    Vec            u, uloc;
    DMDALocalInfo  info;
    const Quad1D   q = gausslegendre[mctx->quaddegree-1];   // from quadrature.h
    PetscReal      xymin[2], xymax[2], hx, hy, **au, x_i, y_j, x, y,
                   ux, uy, W, D,
                   Dminloc = PETSC_INFINITY, Dmaxloc = 0.0, Dmin, Dmax,
                   arealoc = 0.0, area;
    PetscInt       i, j, r, s, tab;
    MPI_Comm       comm;

    PetscCall(SNESGetDM(snes, &da));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(DMGetBoundingBox(info.da,xymin,xymax));
    hx = (xymax[0] - xymin[0]) / (info.mx - 1);
    hy = (xymax[1] - xymin[1]) / (info.my - 1);

    // get the current solution u, with stencil width
    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(DMGetLocalVector(da, &uloc));
    PetscCall(DMGlobalToLocalBegin(da, u, INSERT_VALUES, uloc));
    PetscCall(DMGlobalToLocalEnd(da, u, INSERT_VALUES, uloc));

    // loop over rectangular cells in grid
    PetscCall(DMDAVecGetArrayRead(da,uloc,&au));
    for (j = info.ys; j < info.ys + info.ym; j++) {
        if (j == 0)
            continue;
        y_j = j * hy;  // NE corner of cell is (x_i,y_j)
        for (i = info.xs; i < info.xs + info.xm; i++) {
            if (i == 0)
                continue;
            x_i = i * hx;
            // loop over quadrature points in cell
            for (r = 0; r < q.n; r++) {
                x = x_i - hx + hx * 0.5 * (q.xi[r] + 1);
                for (s = 0; s < q.n; s++) {
                    y = y_j - hy + hy * 0.5 * (q.xi[s] + 1);
                    // gradient of u(x,y) at a quadrature point
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
    PetscCall(DMDAVecRestoreArrayRead(da,uloc,&au));
    PetscCall(DMRestoreLocalVector(da, &uloc));
    arealoc *= hx * hy / 4.0;  // from change of variables formula

    // do global reductions (because could be in parallel)
    PetscCall(PetscObjectGetComm((PetscObject)da,&comm));
    PetscCall(MPI_Allreduce(&arealoc,&area,1,MPIU_REAL,MPIU_SUM,comm));
    PetscCall(MPI_Allreduce(&Dminloc,&Dmin,1,MPIU_REAL,MPIU_MIN,comm));
    PetscCall(MPI_Allreduce(&Dmaxloc,&Dmax,1,MPIU_REAL,MPIU_MAX,comm));

    // report using tabbed (indented) print
    PetscCall(PetscObjectGetTabLevel((PetscObject)snes,&tab));
    PetscCall(PetscViewerASCIIAddTab(PETSC_VIEWER_STDOUT_WORLD,tab));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,
        "area = %.8f; %.4f <= D <= %.4f\n",area,Dmin,Dmax));
    PetscCall(PetscViewerASCIISubtractTab(PETSC_VIEWER_STDOUT_WORLD,tab));
    return 0;
}
