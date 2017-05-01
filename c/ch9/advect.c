static char help[] =
"Time-dependent advection equation, in flux-conservative form, in 2D.\n"
"Option prefix -adv_.  Domain is (-1,1) x (-1,1).  Equation is\n"
"  u_t + div(a(x,y) u) = g(x,y,u).\n"
"Boundary conditions are periodic in x and y.  Cells are grid-point centered.\n"
"Uses flux-limited (non-oscillatory) method-of-lines discretization:\n"
"  none       first-order upwinding (limiter = 0)\n"
"  centered   linear centered fluxes\n"
"  vanleer    van Leer (1974) limiter\n"
"  koren      Koren (1993) limiter [default].\n"
"Solves either of two problems with initial conditions from:\n"
"  straight   Figure 6.2, page 303, in Hundsdorfer & Verwer (2003) [default]\n"
"  rotation   Figure 20.5, page 461, in LeVeque (2002).\n"
"For the former problem, if final time is an integer and velocities are kept\n"
"at default values, then exact solution is available and L1, L2 errors\n"
"are reported.\n\n";

#include <petsc.h>

//STARTCTX
typedef enum {STRAIGHT, ROTATION} ProblemType;
static const char *ProblemTypes[] = {"straight","rotation",
                                     "ProblemType", "", NULL};

typedef enum {STUMP, CONE, BOX} InitialShapeType;
static const char *InitialShapeTypes[] = {"stump", "cone", "box",
                                          "InitialShapeType", "", NULL};

typedef struct {
    ProblemType  problem;
    double       windx, windy;  // x,y velocity if problem==STRAIGHT
    double       (*initialshape)(double,double); // if problem==STRAIGHT
    double       (*limiter)(double);
} AdvectCtx;
//ENDCTX

//STARTINITIALSHAPES
// equal to 1 in a disc of radius 0.2 around (-0.6,-0.6)
static double stump(double x, double y) {
    const double r = PetscSqrtReal((x+0.6)*(x+0.6) + (y+0.6)*(y+0.6));
    return (r < 0.2) ? 1.0 : 0.0;
}

// cone of height 1 of base radius 0.35 centered at (-0.45,0.0)
static double cone(double x, double y) {
    const double r = PetscSqrtReal((x+0.45)*(x+0.45) + y*y);
    return (r < 0.35) ? 1.0 - r / 0.35 : 0.0;
}

// equal to 1 in square of side-length 0.5 (0.1,0.6) x (-0.25,0.25)
static double box(double x, double y) {
    if ((0.1 < x) && (x < 0.6) && (-0.25 < y) && (y < 0.25)) {
        return 1.0;
    } else
        return 0.0;
}

static void* initialshapeptr[] = {&stump, &cone, &box};
//ENDINITIALSHAPES

PetscErrorCode FormInitial(DMDALocalInfo *info, Vec u, AdvectCtx* user) {
    PetscErrorCode ierr;
    int          i, j;
    double       hx, hy, x, y, **au;

    ierr = VecSet(u,0.0); CHKERRQ(ierr);  // clear it first
    ierr = DMDAVecGetArray(info->da, u, &au); CHKERRQ(ierr);
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + (j+0.5) * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + (i+0.5) * hx;
            switch (user->problem) {
                case STRAIGHT:
                    au[j][i] = (*user->initialshape)(x,y);
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

// velocity  a(x,y) = ( a^x(x,y), a^y(x,y) )
static double a_wind(double x, double y, int dir, AdvectCtx* user) {
    switch (user->problem) {
        case STRAIGHT:
            return (dir == 0) ? user->windx : user->windy;
        case ROTATION:
            return (dir == 0) ? 2.0 * y : - 2.0 * x;
        default:
            return 0.0;
    }
}

// source  g(x,y,u)
static double g_source(double x, double y, double u, AdvectCtx* user) {
    return 0.0;
}

//         d g(x,y,u) / d u
static double dg_source(double x, double y, double u, AdvectCtx* user) {
    return 0.0;
}

//STARTLIMITER
/* the centered-space method is a linear (and unlimited) "limiter" */
static double centered(double theta) {
    return 0.5;
}

/* van Leer (1974) limiter is formula (1.11) in section III.1 of
Hundsdorfer & Verwer (2003) */
static double vanleer(double theta) {
    const double abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

/* Koren (1993) limiter is formula (1.7) in section III.1 of
Hundsdorfer & Verwer (2003) */
static double koren(double theta) {
    const double z = (1.0/3.0) + (1.0/6.0) * theta;
    return PetscMax(0.0, PetscMin(1.0, PetscMin(z, theta)));
}

typedef enum {NONE, CENTERED, VANLEER, KOREN} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer","koren",
                                     "LimiterType", "", NULL};
static void* limiterptr[] = {NULL, &centered, &vanleer, &koren};
//ENDLIMITER

/* method-of-lines discretization gives ODE system  u' = G(t,u)
so our finite volume scheme computes
    G_ij = - (fluxE - fluxW)/hx - (fluxN - fluxS)/hy + g(x,y,U_ij)
but only east (E) and north (N) fluxes are computed
*/
//STARTFUNCTION
PetscErrorCode FormRHSFunctionLocal(DMDALocalInfo *info, double t,
        double **au, double **aG, AdvectCtx *user) {
    int         i, j, q;
    double      hx, hy, halfx, halfy, x, y, a,
                u_up, u_dn, u_far, theta, flux;

    // clear G first
    for (j = info->ys; j < info->ys + info->ym; j++)
        for (i = info->xs; i < info->xs + info->xm; i++)
            aG[j][i] = 0.0;
    // fluxes on cell boundaries are traversed in this order:  ,-1-,
    // cell center at * has coordinates (x,y):                 | * 0
    // q = 0,1 is cell boundary index                          '---'
    hx = 2.0 / info->mx;  hy = 2.0 / info->my;
    halfx = hx / 2.0;     halfy = hy / 2.0;
    for (j = info->ys-1; j < info->ys + info->ym; j++) { // note -1 start
        y = -1.0 + (j+0.5) * hy;
        for (i = info->xs-1; i < info->xs + info->xm; i++) { // note -1 start
            x = -1.0 + (i+0.5) * hx;
            if ((i >= info->xs) && (j >= info->ys)) {
                aG[j][i] += g_source(x,y,au[j][i],user);
            }
            for (q = 0; q < 2; q++) {   // get E,N fluxes on cell bdry
                if ((q == 0) && (j < info->ys))  continue;
                if ((q == 1) && (i < info->xs))  continue;
                a = a_wind(x + halfx*(1-q),y + halfy*q,q,user);
                // first-order flux
                u_up = (a >= 0.0) ? au[j][i] : au[j+q][i+(1-q)];
                flux = a * u_up;
                // use flux-limiter
                if (user->limiter != NULL) {
                    // formulas (1.2),(1.3),(1.6); H&V pp 216--217
                    u_dn = (a >= 0.0) ? au[j+q][i+(1-q)] : au[j][i];
                    if (u_dn != u_up) {
                        u_far = (a >= 0.0) ? au[j-q][i-(1-q)]
                                           : au[j+2*q][i+2*(1-q)];
                        theta = (u_up - u_far) / (u_dn - u_up);
                        flux += a * (*user->limiter)(theta)*(u_dn-u_up);
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

PetscErrorCode FormRHSJacobianLocal(DMDALocalInfo *info, double t,
        double **au, Mat J, Mat P, AdvectCtx *user) {
    PetscErrorCode ierr;
    const int   dir[4] = {0, 1, 0, 1},  // use x (0) or y (1) component
                xsh[4]   = { 1, 0,-1, 0},  ysh[4]   = { 0, 1, 0,-1};
    int         i, j, l, nc;
    double      hx, hy, halfx, halfy, x, y, a, v[5];
    MatStencil  col[5],row;

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
            for (l = 0; l < 4; l++) {   // loop over cell boundaries
                a = a_wind(x + halfx*xsh[l],y + halfy*ysh[l],dir[l],user);
                switch (l) {
                    case 0:
                        col[nc].j = j;  col[nc].i = (a >= 0.0) ? i : i+1;
                        v[nc++] = - a / hx;
                        break;
                    case 1:
                        col[nc].j = (a >= 0.0) ? j : j+1;  col[nc].i = i;
                        v[nc++] = - a / hy;
                        break;
                    case 2:
                        col[nc].j = j;  col[nc].i = (a >= 0.0) ? i-1 : i;
                        v[nc++] = a / hx;
                        break;
                    case 3:
                        col[nc].j = (a >= 0.0) ? j-1 : j;  col[nc].i = i;
                        v[nc++] = a / hy;
                        break;
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

// dumps to file; does nothing if string root is empty or NULL
PetscErrorCode dumptobinary(const char* root, const char* append, Vec u) {
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

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    AdvectCtx        user;
    TS               ts;
    DM               da;
    Vec              u;
    DMDALocalInfo    info;
    LimiterType      limiterchoice = KOREN;
    InitialShapeType initialshapechoice = STUMP;
    double           hx, hy, t0, dt, tf;
    char             fileroot[PETSC_MAX_PATH_LEN] = "";
    int              steps;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.problem = STRAIGHT;
    user.windx = 2.0;
    user.windy = 2.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,
           "adv_", "options for advect.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsString("-dumpto","filename root for initial/final state",
           "advect.c",fileroot,fileroot,PETSC_MAX_PATH_LEN,NULL);CHKERRQ(ierr);
//STARTENUMOPTIONS
    ierr = PetscOptionsEnum("-initial",
           "shape of initial condition if problem==straight",
           "advect.c",InitialShapeTypes,
           (PetscEnum)initialshapechoice,(PetscEnum*)&initialshapechoice,NULL); CHKERRQ(ierr);
    user.initialshape = initialshapeptr[initialshapechoice];
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
           "advect.c",LimiterTypes,
           (PetscEnum)limiterchoice,(PetscEnum*)&limiterchoice,NULL); CHKERRQ(ierr);
    user.limiter = limiterptr[limiterchoice];
    ierr = PetscOptionsEnum("-problem","problem type",
           "advect.c",ProblemTypes,
           (PetscEnum)user.problem,(PetscEnum*)&user.problem,NULL); CHKERRQ(ierr);
//ENDENUMOPTIONS
    ierr = PetscOptionsReal("-windx","x component of wind (if problem==straight)",
           "advect.c",user.windx,&user.windx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-windy","y component of wind (if problem==straight)",
           "advect.c",user.windy,&user.windy,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

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

    // grid is cell-centered
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hx = 2.0 / info.mx;  hy = 2.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,
        -1.0+hx/2.0,1.0-hx/2.0,-1.0+hy/2.0,1.0-hy/2.0,0.0,1.0);CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetDM(ts,da); CHKERRQ(ierr);
    ierr = DMDATSSetRHSFunctionLocal(da,INSERT_VALUES,
           (DMDATSRHSFunctionLocal)FormRHSFunctionLocal,&user); CHKERRQ(ierr);
    ierr = DMDATSSetRHSJacobianLocal(da,
           (DMDATSRHSJacobianLocal)FormRHSJacobianLocal,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);  // defaults to -ts_rk_type 3bs
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
    ierr = TSSetInitialTimeStep(ts,0.0,0.1); CHKERRQ(ierr);
    ierr = TSSetDuration(ts,1000000,0.6); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = FormInitial(&info,u,&user); CHKERRQ(ierr);
    ierr = dumptobinary(fileroot,"_initial",u); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&t0); CHKERRQ(ierr);
    ierr = TSGetTimeStep(ts,&dt); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"solving problem '%s' ",
           ProblemTypes[user.problem]); CHKERRQ(ierr);
    if (user.problem == STRAIGHT) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"(initial: %s) ",
           InitialShapeTypes[initialshapechoice]); CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "on %d x %d grid (cells dx=%g x dy=%g),\n"
           "    with t0=%g, initial dt=%g, and '%s' limiter ...\n",
           info.mx,info.my,hx,hy,t0,dt,LimiterTypes[limiterchoice]); CHKERRQ(ierr);

    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    ierr = TSGetTotalSteps(ts,&steps); CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "completed %d steps to time %g\n",steps,tf); CHKERRQ(ierr);
    if ( (user.problem == STRAIGHT) && (PetscAbs(fmod(tf+0.5e-8,1.0)) <= 1.0e-8)
         && (fmod(user.windx,2.0) == 0.0) && (fmod(user.windy,2.0) == 0.0) ) {
        // exact solution is same as initial condition
        Vec    uexact;
        double norms[2];
        ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
        ierr = FormInitial(&info,uexact,&user); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr); // u <- u + (-1.0) uexact
        ierr = VecNorm(u,NORM_1_AND_2,norms); CHKERRQ(ierr);
        norms[0] *= hx * hy;
        norms[1] *= PetscSqrtReal(hx * hy);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "errors |u-uexact|_{1,h} = %.4e, |u-uexact|_{2,h} = %.4e\n",
            norms[0],norms[1]); CHKERRQ(ierr);
        VecDestroy(&uexact);
    }
    ierr = dumptobinary(fileroot,"_final",u); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

