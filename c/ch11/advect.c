static char help[] =
"Time-dependent advection equation, in flux-conservative form, in 2D.\n"
"Option prefix -adv_.  Domain is (-1,1) x (-1,1).  Equation is\n"
"  u_t + div(a(x,y) u) = g(x,y,u).\n"
"Boundary conditions are periodic in x and y.  Cells are grid-point centered.\n"
"Allows flux-limited (non-oscillatory) method-of-lines discretization:\n"
"  none       O(h^1)  first-order upwinding (limiter = 0)\n"
"  centered   O(h^2)  linear centered\n"
"  vanleer    O(h^2)  van Leer (1974) limiter\n"
"  koren      O(h^3)  Koren (1993) limiter [default].\n"
"(There is separate control over the limiter in the residual and in the\n"
"Jacobian.  Only none and centered are implemented for the Jacobian.)\n"
"Solves either of two problems with initial conditions:\n"
"  straight   Figure 6.2, page 303, in Hundsdorfer & Verwer (2003) [default]\n"
"  rotation   Figure 20.5, page 461, in LeVeque (2002).\n"
"For straight, if final time is an integer and velocities are kept at default\n"
"values, then exact solution is known and L1,L2 errors are reported.\n\n";

#include <petsc.h>

//STARTCTX
typedef enum {STUMP, SMOOTH, CONE, BOX} InitialType;
static const char *InitialTypes[] = {"stump", "smooth", "cone", "box",
                                     "InitialType", "", NULL};

typedef enum {NONE, CENTERED, VANLEER, KOREN} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer","koren",
                                     "LimiterType", "", NULL};

typedef enum {STRAIGHT, ROTATION} ProblemType;
static const char *ProblemTypes[] = {"straight","rotation",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType  problem;
    PetscReal    windx, windy,            // x,y velocity in STRAIGHT
                 (*initial_fcn)(PetscReal,PetscReal), // for STRAIGHT
                 (*limiter_fcn)(PetscReal),  // limiter used in RHS
                 (*jac_limiter_fcn)(PetscReal); // used in Jacobian
} AdvectCtx;
//ENDCTX

//STARTINITIAL
// equal to 1 in a disc of radius 0.2 around (-0.6,-0.6)
static PetscReal stump(PetscReal x, PetscReal y) {
    const PetscReal r = PetscSqrtReal((x+0.6)*(x+0.6) + (y+0.6)*(y+0.6));
    return (r < 0.2) ? 1.0 : 0.0;
}

// smooth (C^6) version of stump
static PetscReal smooth(PetscReal x, PetscReal y) {
    const PetscReal r = PetscSqrtReal((x+0.6)*(x+0.6) + (y+0.6)*(y+0.6));
    if (r < 0.2)
        return PetscPowReal(1.0 - PetscPowReal(r / 0.2,6.0),6.0);
    else
        return 0.0;
}

// cone of height 1 of base radius 0.35 centered at (-0.45,0.0)
static PetscReal cone(PetscReal x, PetscReal y) {
    const PetscReal r = PetscSqrtReal((x+0.45)*(x+0.45) + y*y);
    return (r < 0.35) ? 1.0 - r / 0.35 : 0.0;
}

// equal to 1 in square of side-length 0.5 (0.1,0.6) x (-0.25,0.25)
static PetscReal box(PetscReal x, PetscReal y) {
    if ((0.1 < x) && (x < 0.6) && (-0.25 < y) && (y < 0.25))
        return 1.0;
    else
        return 0.0;
}

static void* initialptr[] = {&stump, &smooth, &cone, &box};
//ENDINITIAL

//STARTLIMITERS
/* the centered-space method is linear */
static PetscReal centered(PetscReal theta) {
    return 0.5;
}

/* van Leer (1974) limiter is formula (1.11) in section III.1 of
Hundsdorfer & Verwer (2003) */
static PetscReal vanleer(PetscReal theta) {
    const PetscReal abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

/* Koren (1993) limiter is formula (1.7) in section III.1 of
Hundsdorfer & Verwer (2003) */
static PetscReal koren(PetscReal theta) {
    const PetscReal z = (1.0/3.0) + (1.0/6.0) * theta;
    return PetscMax(0.0, PetscMin(1.0, PetscMin(z, theta)));
}

static void* limiterptr[] = {NULL, &centered, &vanleer, &koren};
//ENDLIMITERS

// velocity  a(x,y) = ( a^x(x,y), a^y(x,y) )
static PetscReal a_wind(PetscReal x, PetscReal y, PetscInt dir, AdvectCtx* user) {
    switch (user->problem) {
        case STRAIGHT:
            return (dir == 0) ? user->windx : user->windy;
        case ROTATION:
            return (dir == 0) ? y : - x;
        default:
            return 0.0;
    }
}

// source  g(x,y,u)
static PetscReal g_source(PetscReal x, PetscReal y, PetscReal u, AdvectCtx* user) {
    return 0.0;
}

//         d g(x,y,u) / d u
static PetscReal dg_source(PetscReal x, PetscReal y, PetscReal u, AdvectCtx* user) {
    return 0.0;
}

extern PetscErrorCode FormInitial(DMDALocalInfo*, Vec, AdvectCtx*);
extern PetscErrorCode DumpBinary(const char*, const char*, Vec);
extern PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo*, PetscReal,
        PetscReal**, PetscReal**, AdvectCtx*);
extern PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo*, PetscReal,
        PetscReal**, Mat, Mat, AdvectCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    TS               ts;
    DM               da;
    Vec              u;
    DMDALocalInfo    info;
    PetscReal        hx, hy, t0, c, dt, tf;
    char             fileroot[PETSC_MAX_PATH_LEN] = "";
    PetscInt         steps;
    PetscBool        oneline = PETSC_FALSE, snesfdset, snesfdcolorset;
    InitialType      initial = STUMP;
    LimiterType      limiter = KOREN, jac_limiter = NONE;
    AdvectCtx        user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.problem = STRAIGHT;
    user.windx = 2.0;
    user.windy = 2.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,
           "adv_", "options for advect.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsString("-dumpto","filename root for binary files with initial/final state",
           "advect.c",fileroot,fileroot,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-initial",
           "shape of initial condition if problem==straight",
           "advect.c",InitialTypes,
           (PetscEnum)initial,(PetscEnum*)&initial,NULL); CHKERRQ(ierr);
    user.initial_fcn = initialptr[initial];
    ierr = PetscOptionsEnum("-limiter",
           "flux-limiter type used in RHS evaluation",
           "advect.c",LimiterTypes,
           (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    user.limiter_fcn = limiterptr[limiter];
    ierr = PetscOptionsEnum("-jac_limiter",
           "flux-limiter type used in Jacobian (of RHS) evaluation",
           "advect.c",LimiterTypes,
           (PetscEnum)jac_limiter,(PetscEnum*)&jac_limiter,NULL); CHKERRQ(ierr);
    user.jac_limiter_fcn = limiterptr[jac_limiter];
    ierr = PetscOptionsEnum("-problem",
           "problem type",
           "advect.c",ProblemTypes,
           (PetscEnum)user.problem,(PetscEnum*)&user.problem,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-oneline",
           "in exact solution cases, show one-line output",
           "advect.c",oneline,&oneline,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windx",
           "x component of wind for problem==straight",
           "advect.c",user.windx,&user.windx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windy",
           "y component of wind for problem==straight",
           "advect.c",user.windy,&user.windy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = PetscOptionsHasName(NULL,NULL,"-snes_fd",&snesfdset); CHKERRQ(ierr);
    ierr = PetscOptionsHasName(NULL,NULL,"-snes_fd_color",&snesfdcolorset); CHKERRQ(ierr);
    if (snesfdset || snesfdcolorset) {
        user.jac_limiter_fcn = NULL;
        jac_limiter = 5;   // corresponds to empty string
    }

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
               DMDA_STENCIL_STAR,              // no diagonal differencing
               5,5,PETSC_DECIDE,PETSC_DECIDE,  // default to hx=hx=0.2 grid
                                               //   (mx=my=5 allows -snes_fd_color)
               1, 2,                           // d.o.f & stencil width
               NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 2.0 / info.mx;  hy = 2.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,    // grid is cell-centered
        -1.0+hx/2.0,1.0-hx/2.0,-1.0+hy/2.0,1.0-hy/2.0,0.0,1.0);CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSJacobianLocal(da,
           (DMDATSRHSJacobianLocal)FormRHSJacobianLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);  // defaults to -ts_rk_type 3bs

    // time axis: use CFL number of 0.5 to set initial time step, but note
    //            most methods adapt anyway
    if (user.problem == STRAIGHT)
        c = PetscMax(PetscAbsReal(user.windx)/hx, PetscAbsReal(user.windy)/hy);
    else
        c = PetscMax(1.0/hx, 1.0/hy);
    dt = 0.5 / c;
    ierr = TSSetTime(ts,0.0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,0.6); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = FormInitial(&info,u,&user); CHKERRQ(ierr);
    ierr = DumpBinary(fileroot,"_initial",u); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);

    if (!oneline) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
               "solving problem %s with %s initial state on %d x %d grid,\n"
               "    cells dx=%g x dy=%g, limiter = %s, and jac_limiter = %s ...\n",
               ProblemTypes[user.problem],InitialTypes[initial],info.mx,info.my,
               hx,hy,LimiterTypes[limiter],LimiterTypes[jac_limiter]); CHKERRQ(ierr);
    }

    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = DumpBinary(fileroot,"_final",u); CHKERRQ(ierr);

    if (!oneline) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
                "completed %d steps to time %g\n",steps,tf); CHKERRQ(ierr);
    }

    if ( (user.problem == STRAIGHT) && (PetscAbs(fmod(tf+0.5e-8,1.0)) <= 1.0e-8)
         && (fmod(user.windx,2.0) == 0.0) && (fmod(user.windy,2.0) == 0.0) ) {
        // exact solution is same as initial condition
        Vec    uexact;
        PetscReal norms[2];
        ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
        ierr = FormInitial(&info,uexact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr); // u <- u + (-1.0) uexact
        ierr = VecNorm(u,NORM_1_AND_2,norms); CHKERRQ(ierr);
        norms[0] *= hx * hy;
        norms[1] *= PetscSqrtReal(hx * hy);
        VecDestroy(&uexact);
        if (oneline) {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "%s,%s,%s,%d,%d,%g,%g,%d,%g,%.4e,%.4e\n",
                ProblemTypes[user.problem],InitialTypes[initial],
                LimiterTypes[limiter],info.mx,info.my,hx,hy,steps,tf,
                norms[0],norms[1]); CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "errors |u-uexact|_{1,h} = %.4e, |u-uexact|_{2,h} = %.4e\n",
                norms[0],norms[1]); CHKERRQ(ierr);
        }
    }

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    return PetscFinalize();
}

PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, AdvectCtx* user) {
    PetscErrorCode ierr;
    PetscInt   i, j;
    PetscReal  hx, hy, x, y, **au;

    ierr = VecSet(u,0.0); CHKERRQ(ierr);  // clear it first
    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + (j+0.5) * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + (i+0.5) * hx;
            switch (user->problem) {
                case STRAIGHT:
                    au[j][i] = (*user->initial_fcn)(x,y);
                    break;
                case ROTATION:
                    au[j][i] = cone(x,y) + box(x,y);
                    break;
                default:
                    SETERRQ(PETSC_COMM_WORLD,1,"invalid user->problem\n");
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au); CHKERRQ(ierr);
    return 0;
}

// dumps to file; does nothing if string root is empty or NULL
PetscErrorCode DumpBinary(const char* root, const char* append, Vec u) {
    PetscErrorCode ierr;
    if ((root) && (strlen(root) > 0)) {
        PetscViewer  viewer;
        char         filename[PETSC_MAX_PATH_LEN] = "";
        sprintf(filename,"%s%s.dat",root,append);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "writing PETSC binary file %s ...\n",filename); CHKERRQ(ierr);
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,
            FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
        ierr = VecView(u,viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
    }
    return 0;
}

/* method-of-lines discretization gives ODE system  u' = G(t,u)
so our finite volume scheme computes
    G_ij = - (fluxE - fluxW)/hx - (fluxN - fluxS)/hy + g(x,y,U_ij)
but only east (E) and north (N) fluxes are computed
*/
//STARTFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal **au, PetscReal **aG, AdvectCtx *user) {
    PetscInt   i, j, q, dj, di;
    PetscReal  hx, hy, halfx, halfy, x, y, a,
               u_up, u_dn, u_far, theta, flux;

    // clear G first
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            aG[j][i] = 0.0;
    // fluxes on cell boundaries are traversed in E,N order with indices
    // q=0 for E and q=1 for N; cell center has coordinates (x,y)
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys-1; j < info->ys + info->ym; j++) { // note -1 start
        y = -1.0 + (j+0.5) * hy;
        for (i = info->xs-1; i < info->xs + info->xm; i++) { // -1 start
            x = -1.0 + (i+0.5) * hx;
            if ((i >= info->xs) && (j >= info->ys)) {
                aG[j][i] += g_source(x,y,au[j][i],user);
            }
            for (q = 0; q < 2; q++) {   // E (q=0) and N (q=1) bdry fluxes
                if (q == 0 && j < info->ys)  continue;
                if (q == 1 && i < info->xs)  continue;
                di = 1 - q;
                dj = q;
                a = a_wind(x + halfx*di,y + halfy*dj,q,user);
                // first-order flux
                u_up = (a >= 0.0) ? au[j][i] : au[j+dj][i+di];
                flux = a * u_up;
                // use flux-limiter
                if (user->limiter_fcn != NULL) {
                    // formulas (1.2),(1.3),(1.6); H&V pp 216--217
                    u_dn = (a >= 0.0) ? au[j+dj][i+di] : au[j][i];
                    if (u_dn != u_up) {
                        u_far = (a >= 0.0) ? au[j-dj][i-di]
                                           : au[j+2*dj][i+2*di];
                        theta = (u_up - u_far) / (u_dn - u_up);
                        flux += a * (*user->limiter_fcn)(theta)
                                  * (u_dn-u_up);
                    }
                }
                // update owned G_ij on both sides of computed flux
                if (q == 0) {
                    if (i >= info->xs)
                        aG[j][i]   -= flux / hx;
                    if (i+1 < info->xs + info->xm)
                        aG[j][i+1] += flux / hx;
                } else {
                    if (j >= info->ys)
                        aG[j][i]   -= flux / hy;
                    if (j+1 < info->ys + info->ym)
                        aG[j+1][i] += flux / hy;
                }
            }
        }
    }
    return 0;
}
//ENDFUNCTION

PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo *info, PetscReal t,
        PetscReal **au, Mat J, Mat P, AdvectCtx *user) {
    PetscErrorCode ierr;
    const PetscInt  dir[4] = { 0, 1, 0, 1},  // use x (0) or y (1) component
                    xsh[4] = { 1, 0,-1, 0},  ysh[4]   = { 0, 1, 0,-1};
    PetscInt        i, j, l, nc;
    PetscReal       hx, hy, halfx, halfy, x, y, a, v[9];
    MatStencil      col[9],row;

    ierr = MatZeroEntries(P); CHKERRQ(ierr);
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys; j < info->ys+info->ym; j++) {
        y = -1.0 + (j+0.5) * hy;
        row.j = j;
        for (i = info->xs; i < info->xs+info->xm; i++) {
            x = -1.0 + (i+0.5) * hx;
            row.i = i;
            col[0].j = j;  col[0].i = i;
            v[0] = dg_source(x,y,au[j][i],user);
            nc = 1;
            for (l = 0; l < 4; l++) {   // loop over cell boundaries: E, N, W, S
                a = a_wind(x + halfx*xsh[l],y + halfy*ysh[l],dir[l],user);
                if (user->jac_limiter_fcn == NULL) {
                    // Jacobian is from upwind fluxes
                    switch (l) {
                        case 0:
                            col[nc].j = j;
                            col[nc].i = (a >= 0.0) ? i : i+1;
                            v[nc++] = - a / hx;
                            break;
                        case 1:
                            col[nc].j = (a >= 0.0) ? j : j+1;
                            col[nc].i = i;
                            v[nc++] = - a / hy;
                            break;
                        case 2:
                            col[nc].j = j;
                            col[nc].i = (a >= 0.0) ? i-1 : i;
                            v[nc++] = a / hx;
                            break;
                        case 3:
                            col[nc].j = (a >= 0.0) ? j-1 : j;
                            col[nc].i = i;
                            v[nc++] = a / hy;
                            break;
                    }
                } else if (user->jac_limiter_fcn == &centered) {
                    // Jacobian is from centered fluxes
                    switch (l) {
                        case 0:
                            col[nc].j = j;  col[nc].i = i;    v[nc++] = - a / (2.0*hx);
                            col[nc].j = j;  col[nc].i = i+1;  v[nc++] = - a / (2.0*hx);
                            break;
                        case 1:
                            col[nc].j = j;    col[nc].i = i;  v[nc++] = - a / (2.0*hy);
                            col[nc].j = j+1;  col[nc].i = i;  v[nc++] = - a / (2.0*hy);
                            break;
                        case 2:
                            col[nc].j = j;  col[nc].i = i-1;  v[nc++] = a / (2.0*hx);
                            col[nc].j = j;  col[nc].i = i;    v[nc++] = a / (2.0*hx);
                            break;
                        case 3:
                            col[nc].j = j-1;  col[nc].i = i;  v[nc++] = a / (2.0*hy);
                            col[nc].j = j;    col[nc].i = i;  v[nc++] = a / (2.0*hy);
                            break;
                    }
                } else {
                    SETERRQ(PETSC_COMM_WORLD,1,"only Jacobian cases none|centered are implemented\n");
                }
            }
            ierr = MatSetValuesStencil(P,1,&row,nc,col,v,ADD_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

