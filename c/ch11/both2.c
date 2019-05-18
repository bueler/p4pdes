static char help[] =
"Solves 2D advection-diffusion problems using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -b2_.\n"
"Equation:\n"
"    - eps Laplacian u + Div (a(x,y) u) = f(x,y),\n"
"where the (vector) wind a(x,y) and (scalar) source f(x,y) are given smooth\n"
"functions.  The domain is S = (-1,1)^2 with Dirichlet boundary conditions:\n"
"    u = g(x,y) on boundary S\n"
"where g(x,y) is a given smooth function.  Problems include: NOWIND, LAYER,\n"
"INTERNAL, and GLAZE.  The first of these has a=0 while the last three are\n"
"Examples 6.1.1, 6.1.3, and 6.1.4 in Elman et al (2014), respectively.\n"
"Advection can be discretized by first-order upwinding (none), centered, or a\n"
"van Leer limiter scheme.  Option allows using none limiter on all grids\n"
"but the finest in geometric multigrid.\n\n";

#include <petsc.h>

typedef enum {NONE, CENTERED, VANLEER} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer",
                                     "LimiterType", "", NULL};

static double centered(double theta) {
    return 0.5;
}

static double vanleer(double theta) {
    const double abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

static void* limiterptr[] = {NULL, &centered, &vanleer};

typedef enum {NOWIND, LAYER, INTERNAL, GLAZE} ProblemType;
static const char *ProblemTypes[] = {"nowind", "layer", "internal", "glaze",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType problem;
    double      eps;                              // diffusion eps > 0
    double      (*limiter_fcn)(double),
                (*f_fcn)(double, double, void*),  // source
                (*g_fcn)(double, double, void*);  // boundary condition
    PetscBool   none_on_down;                     // use none limiter except
    int         mx_fine, my_fine;                 //    on finest grid
} AdCtx;

// used for source functions
static double zero(double x, double y, void *user) {
    return 0.0;
}

// problem NOWIND: same problem as ./fish -fsh_problem manuexp
static double nowind_u(double x, double y, void *user) {  // exact solution
    return - x * exp(y);
}

static double nowind_f(double x, double y, void *user) {
    AdCtx* usr = (AdCtx*)user;
    return usr->eps * x * exp(y);
}

static double nowind_g(double x, double y, void *user) {
    return nowind_u(x,y,user);
}

// problem LAYER:  Elman page 237, Example 6.1.1
static double layer_u(double x, double y, AdCtx *user) {  // exact solution
    AdCtx* usr = (AdCtx*)user;
    return x * (1.0 - exp((y-1) / usr->eps)) / (1.0 - exp(- 2.0 / usr->eps));
}

static double layer_g(double x, double y, void *user) {
    return layer_u(x,y,user);
}

// problem INTERNAL:  Elman page 239, Example 6.1.3
static double internal_g(double x, double y, void *user) {
    if (y > 2*x - 1)
       return 1.0;   // along x=1 and (y=-1 & 0 < x < 1) boundaries
    else
       return 0.0;
}

// problem GLAZE:  Elman page 240, Example 6.1.4
static double glaze_g(double x, double y, void *user) {
    if (x > 0.0 && y < x && y > -x)
       return 1.0;   // along x=1 boundary
    else
       return 0.0;
}

static void* uexptr[] = {&nowind_u, &layer_u, NULL,        NULL};
static void* fptr[]   = {&nowind_f, &zero,    &zero,       &zero};
static void* gptr[]   = {&nowind_g, &layer_g, &internal_g, &glaze_g};

/* This vector function returns q=0,1 component.  It is used in
   FormFunctionLocal() to get a(x,y). */
static double wind_a(double x, double y, int q, AdCtx *user) {
    switch (user->problem) {
        case NOWIND:
            return 0.0;
        case LAYER:
            return (q == 0) ? 1.0 : 0.0;
        case INTERNAL:
            return (q == 0) ? -0.5 : PetscSqrtReal(3.0)/2.0;
        case GLAZE:
            return (q == 0) ? 2.0*y*(1.0-x*x) : -2.0*x*(1.0-y*y);
        default:
            return 1.0e308 * 100.0;  // cause overflow
    }
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, 
                                 double (*)(double, double, void*), Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double***,
                                        double***, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         hx, hy, err2;
    DMDALocalInfo  info;
    double         (*uexact_fcn)(double, double, void*);
    LimiterType    limiter = NONE;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 0.01;
    user.none_on_down = PETSC_FALSE;
    user.problem = LAYER;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"b2_",
               "both2 (2D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","positive diffusion coefficient",
               "both2.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "both2.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-none_on_down",
               "on grids coarser than the finest, disregard limiter choices and use none",
               "both2.c",user.none_on_down,&(user.none_on_down),NULL);
               CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem","problem type",
               "both2.c",ProblemTypes,
               (PetscEnum)(user.problem),(PetscEnum*)&(user.problem),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];
    uexact_fcn = uexptr[user.problem];
    user.f_fcn = fptr[user.problem];
    user.g_fcn = gptr[user.problem];

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
        3,3,                          // default to hx=hy=1 grid
        PETSC_DECIDE,PETSC_DECIDE,
        1,2,                          // d.o.f, stencil width
        NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    user.mx_fine = info.mx;
    user.my_fine = info.my;
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(snes,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d grid (problem = %s, eps = %g, limiter = %s)\n",
         info.mx,info.my,ProblemTypes[user.problem],user.eps,LimiterTypes[limiter]);
         CHKERRQ(ierr);

    if (uexact_fcn != NULL) {
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormUExact(&info,&user,uexact_fcn,u_exact); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
        ierr = VecNorm(u,NORM_2,&err2); CHKERRQ(ierr);
        hx = 2.0 / (info.mx - 1);
        hy = 2.0 / (info.my - 1);
        err2 *= PetscSqrtReal(hx * hy);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
             "numerical error:  |u-uexact|_2 = %.4e\n",err2); CHKERRQ(ierr);
        ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
    }

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormUExact(DMDALocalInfo *info, AdCtx *usr,
                          double (*uexact)(double, double, void*), Vec uex) {
    PetscErrorCode  ierr;
    int          i, j;
    double       hx, hy, x, y, **auex;

    if (uexact == NULL) {
        SETERRQ(PETSC_COMM_WORLD,1,"exact solution not available");
    }
    hx = 2.0 / (info->mx - 1);
    hy = 2.0 / (info->my - 1);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = -1.0 + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = -1.0 + i * hx;
            auex[j][i] = (*uexact)(x,y,usr);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}

/* compute residuals:
     F_ij = (- eps Laplacian u + Div (a(x,y) u) - f(x,y)) * hx * hy
at boundary points:
     F_ij = c (u - g(x,y))                                          */
FIXME
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double ***au,
                                 double ***aF, AdCtx *usr) {
    int          i, j, k, p, di, dj, dk;
    double       hx, hy, hz, halfx, halfy, halfz, hx2, hy2, hz2, scF, scDir,
                 x, y, z,
                 uu, uE, uW, uT, uB, uxx, uyy, uzz,
                 ap, flux, u_up, u_dn, u_far, theta;
    PetscBool    allowdeep;

    hx = 2.0 / (info->mx - 1);
    hy = 2.0 / info->my;
    hz = 2.0 / (info->mz - 1);
    halfx = hx / 2.0;
    halfy = hy / 2.0;
    halfz = hz / 2.0;
    hx2 = hx * hx;
    hy2 = hy * hy;
    hz2 = hz * hz;
    // scale as in fish -fsh_dim 3
    scF = hx * hy * hz;
    scDir = scF * usr->eps * 2.0 * (1.0 / hx2 + 1.0 / hy2 + 1.0 / hz2);

    // for each owned cell, compute non-advective parts of residual at
    // cell center
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + (j + 0.5) * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * hx;
                if (i == info->mx-1) {   // x=1 boundary has nonhomo. Dirichlet
                    aF[k][j][i] = scDir * (au[k][j][i] - (*usr->b_fcn)(y,z,usr));
                } else if (i == 0 || k == 0 || k == info->mz-1) {
                    aF[k][j][i] = scDir * au[k][j][i];
                } else {
                    uu = au[k][j][i];
                    uE = (i == info->mx-2) ? (*usr->b_fcn)(y,z,usr) : au[k][j][i+1];
                    uW = (i == 1)          ?                    0.0 : au[k][j][i-1];
                    uT = (k == info->mz-2) ?                    0.0 : au[k+1][j][i];
                    uB = (k == 1)          ?                    0.0 : au[k-1][j][i];
                    uxx = (uW - 2.0 * uu + uE) / hx2;
                    uyy = (au[k][j-1][i] - 2.0 * uu + au[k][j+1][i]) / hy2;
                    uzz = (uB - 2.0 * uu + uT) / hz2;
                    aF[k][j][i] = scF * (- usr->eps * (uxx + uyy + uzz)
                                         - (*usr->g_fcn)(x,y,z,usr));
                }
            }
        }
    }

    // for each E,N,T face of an owned cell, compute flux at the face center
    // and then add that to the correct residual
    // note -1 starts to get W,S,B faces of owned cells living on ownership
    // boundaries for i,j,k resp.
    for (k=info->zs-1; k<info->zs+info->zm; k++) {
        z = -1.0 + k * hz;
        for (j=info->ys-1; j<info->ys+info->ym; j++) {
            y = -1.0 + (j + 0.5) * hy;
            for (i=info->xs-1; i<info->xs+info->xm; i++) {
                x = -1.0 + i * hx;
                // consider cell centered at (x,y,z) and (i,j,k) ...
                // if cell center is on x=1 or z=1 boundaries then we
                //   do not need to compute any face-center fluxes
                if (i == info->mx-1 || k == info->mz-1)
                    continue;
                // traverse E,N,T cell face center points for flux
                //   contributions (E,N,T correspond to p=0,1,2)
                for (p = 0; p < 3; p++) {
                    if (((i == 0 || i < info->xs) && p != 0) || (i < 0)) // skip N,T
                        continue;
                    if  (j < info->ys             && p != 1)             // skip E,T
                        continue;
                    if (((k == 0 || k < info->zs) && p != 2) || (k < 0)) // skip E,N
                        continue;
                    // location on other side of face
                    di = (p == 0) ? 1 : 0;
                    dj = (p == 1) ? 1 : 0;
                    dk = (p == 2) ? 1 : 0;
                    // get pth component of wind and first-order upwind flux
                    ap = wind_a(x + halfx * di, y + halfy * dj, z + halfz * dk,
                                p,usr);
                    if (ap >= 0.0) {
                        u_up = au[k][j][i];
                    } else {
                        if (i+di == info->mx-1) {
                            u_up = (*usr->b_fcn)(y,z,usr);
                        } else if (k+dk == info->mz-1) {
                            u_up = 0.0;
                        } else {
                            u_up = au[k+dk][j+dj][i+di];
                        }
                    }
                    flux = ap * u_up;
                    // flux correction if have limiter and not near boundaries
                    if (usr->limiter_fcn != NULL) {
                        allowdeep = (   (p == 0 && i > 0 && i < info->mx-2)
                                     || (p == 1)
                                     || (p == 2 && k > 0 && k < info->mz-2) );
                        if (allowdeep) {
                            // compute flux correction from high-order formula with psi(theta)
                            u_dn = (ap >= 0.0) ? au[k+dk][j+dj][i+di] : au[k][j][i];
                            if (u_dn != u_up) {
                                u_far = (ap >= 0.0) ? au[k-dk][j-dj][i-di]         // FIXME uminus could be bdry
                                                    : au[k+2*dk][j+2*dj][i+2*di];  // FIXME uplus2 could be bdry
                                theta = (u_up - u_far) / (u_dn - u_up);
                                flux += ap * (*usr->limiter_fcn)(theta) * (u_dn - u_up);
                            }
                        }
                    }
                    // update non-boundary and owned F_ijk on both sides of computed flux
                    switch (p) {
                        case 0:  // flux at E
                            if (i > 0)
                                aF[k][j][i]   += scF * flux / hx;  // flux out of i,j,k at E
                            if (i+1 < info->mx && i+1 < info->xs + info->xm)
                                aF[k][j][i+1] -= scF * flux / hx;  // flux into i+1,j,k at W
                            break;
                        case 1:  // flux at N
                            if (j >= info->ys)
                                aF[k][j][i]   += scF * flux / hy;
                            if (j+1 < info->ys + info->ym)
                                aF[k][j+1][i] -= scF * flux / hy;
                            break;
                        case 3:  // flux at T
                            if (k > 0)
                                aF[k][j][i]   += scF * flux / hz;
                            if (k+1 < info->mz && k+1 < info->zs + info->zm)
                                aF[k+1][j][i] -= scF * flux / hz;
                            break;
                    }
                }
            }
        }
    }

    return 0;
}

