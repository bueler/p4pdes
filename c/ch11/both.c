static char help[] =
"Solves 2D advection-diffusion problems using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -bth_.\n"
"Equation:\n"
"    - eps Laplacian u + Div (a(x,y) u) = g(x,y),\n"
"where the (vector) wind a(x,y) and (scalar) source g(x,y) are given smooth\n"
"functions.  The domain is S = (-1,1)^2 with Dirichlet boundary conditions:\n"
"    u = b(x,y) on boundary S\n"
"where b(x,y) is a given smooth function.  Problems include: NOWIND, LAYER,\n"
"and GLAZE.  The first of these has a=0 while LAYER and GLAZE are\n"
"Examples 6.1.1 and 6.1.4 in Elman et al (2014), respectively.\n"
"Advection can be discretized by first-order upwinding (none), centered, or a\n"
"van Leer limiter scheme.  Option allows switching to none limiter on all grids\n"
"for which the mesh Peclet P^h exceeds a threshold (default: 1).\n\n";

#include <petsc.h>

typedef enum {NONE, CENTERED, VANLEER} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer",
                                     "LimiterType", "", NULL};

static PetscReal centered(PetscReal theta) {
    return 0.5;
}

static PetscReal vanleer(PetscReal theta) {
    const PetscReal abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);   // 4 flops
}

typedef PetscReal (*LimiterFcn)(PetscReal);

static LimiterFcn limiterptr[] = {NULL, &centered, &vanleer};

typedef enum {NOWIND, LAYER, GLAZE} ProblemType;
static const char *ProblemTypes[] = {"nowind", "layer", "glaze",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType  problem;
    PetscReal    eps,                              // diffusion eps > 0
                 a_scale,                          // scale for wind
                 peclet_threshold,
                 (*limiter_fcn)(PetscReal),
                 (*g_fcn)(PetscReal, PetscReal, void*),  // right-hand-side source
                 (*b_fcn)(PetscReal, PetscReal, void*);  // boundary condition
    PetscBool    none_on_peclet,                   // if true use none limiter when P^h > threshold
                 small_peclet_achieved;            // true if on finest grid P^h <= threshold
} AdCtx;

// used for source functions
static PetscReal zero(PetscReal x, PetscReal y, void *user) {
    return 0.0;
}

// problem NOWIND: same problem as ./fish -fsh_problem manuexp
static PetscReal nowind_u(PetscReal x, PetscReal y, void *user) {  // exact solution
    return - x * PetscExpReal(y);
}

static PetscReal nowind_g(PetscReal x, PetscReal y, void *user) {
    AdCtx* usr = (AdCtx*)user;
    return usr->eps * x * PetscExpReal(y);
}

static PetscReal nowind_b(PetscReal x, PetscReal y, void *user) {
    return nowind_u(x,y,user);
}

// problem LAYER:  Elman page 237, Example 6.1.1
static PetscReal layer_u(PetscReal x, PetscReal y, void *user) {  // exact solution
    AdCtx* usr = (AdCtx*)user;
    return x * (1.0 - PetscExpReal((y-1) / usr->eps))
             / (1.0 - PetscExpReal(- 2.0 / usr->eps));
}

static PetscReal layer_b(PetscReal x, PetscReal y, void *user) {
    return layer_u(x,y,user);
}

// problem GLAZE:  Elman page 240, Example 6.1.4
static PetscReal glaze_b(PetscReal x, PetscReal y, void *user) {
    if (x > 0.0 && y < x && y > -x)
       return 1.0;   // along x=1 boundary
    else
       return 0.0;
}

typedef PetscReal (*PointwiseFcn)(PetscReal,PetscReal,void*);

static PointwiseFcn uexptr[] = {&nowind_u, &layer_u, NULL};
static PointwiseFcn gptr[]   = {&nowind_g, &zero,    &zero};
static PointwiseFcn bptr[]   = {&nowind_b, &layer_b, &glaze_b};

/* This vector function returns q=0,1 component.  It is used in
   FormFunctionLocal() to get a(x,y). */
static PetscReal wind_a(PetscReal x, PetscReal y, PetscInt q, AdCtx *user) {
    switch (user->problem) {
        case NOWIND:
            return 0.0;
        case LAYER:
            return (q == 0) ? 0.0 : 1.0;
        case GLAZE:
            return (q == 0) ? 2.0*y*(1.0-x*x) : -2.0*x*(1.0-y*y);
        default:
            return NAN;
    }
}

extern PetscErrorCode FormUExact(DMDALocalInfo*,AdCtx*, 
                                 PetscReal (*)(PetscReal, PetscReal, void*),Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,PetscReal**,PetscReal**,AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u;
    DMDALocalInfo  info;
    PointwiseFcn   uexact_fcn;
    LimiterType    limiter = NONE;
    PetscBool      init_exact = PETSC_FALSE;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 0.005;
    user.none_on_peclet = PETSC_FALSE;
    user.small_peclet_achieved = PETSC_FALSE;
    user.problem = LAYER;
    user.a_scale = 1.0;   // this could be made dependent on problem
    user.peclet_threshold = 1.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"bth_",
               "both (2D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","positive diffusion coefficient",
               "both.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-init_exact","use exact solution for initialization",
               "both.c",init_exact,&init_exact,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "both.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-none_on_peclet",
               "on coarse grids such that mesh Peclet P^h exceeds threshold, switch to none limiter",
               "both.c",user.none_on_peclet,&(user.none_on_peclet),NULL);
               CHKERRQ(ierr);
    ierr = PetscOptionsReal("-peclet_threshold",
               "if mesh Peclet P^h is above this value, switch to none (used with -bth_none_on_peclet)",
               "both.c",user.peclet_threshold,&(user.peclet_threshold),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem","problem type",
               "both.c",ProblemTypes,
               (PetscEnum)(user.problem),(PetscEnum*)&(user.problem),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];
    uexact_fcn = uexptr[user.problem];
    user.g_fcn = gptr[user.problem];
    user.b_fcn = bptr[user.problem];

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
        3,3,                          // default to hx=hy=1 grid
        PETSC_DECIDE,PETSC_DECIDE,
        1,2,                          // d.o.f, stencil width
        NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    if (user.problem == NOWIND) {
        ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,1.0); CHKERRQ(ierr);
    } else {
        ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0); CHKERRQ(ierr);
    }
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(snes,&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&u_initial); CHKERRQ(ierr);
    if (init_exact) {
        ierr = FormUExact(&info,&user,uexact_fcn,u_initial); CHKERRQ(ierr);
    } else {
        ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    }
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    if (user.none_on_peclet && !user.small_peclet_achieved) {
         ierr = PetscPrintf(PETSC_COMM_WORLD,
             "WARNING: -bth_none_on_peclet set but finest grid used NONE limiter\n");
             CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d grid (problem = %s, eps = %g, limiter = %s)\n",
         info.mx,info.my,ProblemTypes[user.problem],user.eps,LimiterTypes[limiter]);
         CHKERRQ(ierr);

    if (uexact_fcn != NULL) {
        Vec     u_exact;
        PetscReal  xymin[2], xymax[2], hx, hy, err2;
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormUExact(&info,&user,uexact_fcn,u_exact); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
        ierr = VecNorm(u,NORM_2,&err2); CHKERRQ(ierr);
        ierr = DMGetBoundingBox(da_after,xymin,xymax); CHKERRQ(ierr);
        hx = (xymax[0] - xymin[0]) / (info.mx - 1);
        hy = (xymax[1] - xymin[1]) / (info.my - 1);
        err2 *= PetscSqrtReal(hx * hy);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
             "numerical error:  |u-uexact|_2 = %.4e\n",err2); CHKERRQ(ierr);
        ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
    }

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormUExact(DMDALocalInfo *info, AdCtx *usr,
                          PetscReal (*uexact)(PetscReal, PetscReal, void*), Vec uex) {
    PetscErrorCode  ierr;
    PetscInt        i, j;
    PetscReal       xymin[2], xymax[2], hx, hy, x, y, **auex;

    if (uexact == NULL) {
        SETERRQ(PETSC_COMM_WORLD,1,"exact solution not available");
    }
    ierr = DMGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = xymin[0] + i * hx;
            auex[j][i] = (*uexact)(x,y,usr);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}

/* compute residuals:
     F_ij = hx * hy * (- eps Laplacian u + Div (a(x,y) u) - g(x,y))
at boundary points:
     F_ij = c (u - b(x,y))
note compass notation for faces of cells; advection scheme evaluates flux at
each E,N face once and then includes it in two residuals:
     N
  -------
  |     |
W |  *  | E
  |     |
  -------
     S
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **au,
                                 PetscReal **aF, AdCtx *usr) {
    PetscErrorCode ierr;
    PetscInt        i, j, p;
    PetscReal       xymin[2], xymax[2], hx, hy, Ph, hx2, hy2, scF, scBC,
                    x, y, uE, uW, uN, uS, uxx, uyy,
                    ap, flux, u_up, u_dn, u_far, theta;
    PetscReal       (*limiter)(PetscReal);
    PetscBool       iowned, jowned, ip1owned, jp1owned;
    PetscLogDouble  ff;

    ierr = DMGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    limiter = usr->limiter_fcn;
    if (usr->none_on_peclet) {
        Ph = usr->a_scale * PetscMax(hx,hy) / usr->eps;  // mesh Peclet number
        if (Ph > usr->peclet_threshold)
            limiter = NULL;
        else
            usr->small_peclet_achieved = PETSC_TRUE;
    }
    hx2 = hx * hx;
    hy2 = hy * hy;
    scF = hx * hy;  // scale residuals
    scBC = scF * usr->eps * 2.0 * (1.0 / hx2 + 1.0 / hy2); // scale b.c. residuals

    // for owned cells, compute non-advective parts of residual at cell center
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i == 0 || i == info->mx-1 || j == 0 || j == info->my-1) {
                aF[j][i] = scBC * (au[j][i] - (*usr->b_fcn)(x,y,usr));
            } else {
                uE = (i+1 == info->mx-1) ? (*usr->b_fcn)(xymax[0],y,usr) : au[j][i+1];
                uW = (i-1 == 0)          ? (*usr->b_fcn)(xymin[0],y,usr) : au[j][i-1];
                uxx = (uE - 2.0 * au[j][i] + uW) / hx2;
                uN = (j+1 == info->my-1) ? (*usr->b_fcn)(x,xymax[1],usr) : au[j+1][i];
                uS = (j-1 == 0)          ? (*usr->b_fcn)(x,xymin[1],usr) : au[j-1][i];
                uyy = (uN - 2.0 * au[j][i] + uS) / hy2;
                aF[j][i] = scF * (- usr->eps * (uxx + uyy) - (*usr->g_fcn)(x,y,usr));
            }
        }
    }
    ierr = PetscLogFlops(14.0*info->xm*info->ym); CHKERRQ(ierr);

    // for each E,N face of an *owned* cell at (x,y) and (i,j), compute flux at
    //     the face center and then add that to the correct residual
    // note start offset of -1; gets W,S faces of owned cells living on ownership
    //     boundaries for i,j resp.
    // there are (xm+1)*(ym+1)*2 fluxes to evaluate
    for (j=info->ys-1; j<info->ys+info->ym; j++) {
        y = xymin[1] + j * hy;
        // if y<0 or y=1 at cell center then no need to compute *any* E,N face-center fluxes
        if (j < 0 || j == info->my-1)
            continue;
        for (i=info->xs-1; i<info->xs+info->xm; i++) {
            x = xymin[0] + i * hx;
            // if x<0 or x=1 at cell center then no need to compute *any* E,N face-center fluxes
            if (i < 0 || i == info->mx-1)
                continue;
            // get E (p=0) and N (p=1) cell face-center flux contributions
            for (p = 0; p < 2; p++) {
                // get pth component of wind and locations determined by wind direction
                ap = (p == 0) ? wind_a(x+hx/2.0,y,p,usr) : wind_a(x,y+hy/2.0,p,usr);
                if (p == 0)
                    if (ap >= 0.0) {
                        u_up  = (i == 0)            ? (*usr->b_fcn)(xymin[0],y,usr) : au[j][i];
                        u_dn  = (i+1 == info->mx-1) ? (*usr->b_fcn)(xymax[0],y,usr) : au[j][i+1];
                        u_far = (i-1 <= 0)          ? (*usr->b_fcn)(xymin[0],y,usr) : au[j][i-1];
                    } else {
                        u_up  = (i+1 == info->mx-1) ? (*usr->b_fcn)(xymax[0],y,usr) : au[j][i+1];
                        u_dn  = (i == 0)            ? (*usr->b_fcn)(xymin[0],y,usr) : au[j][i];
                        u_far = (i+2 >= info->mx-1) ? (*usr->b_fcn)(xymax[0],y,usr) : au[j][i+2];
                    }
                else  // p == 1
                    if (ap >= 0.0) {
                        u_up  = (j == 0)            ? (*usr->b_fcn)(x,xymin[1],usr) : au[j][i];
                        u_dn  = (j+1 == info->my-1) ? (*usr->b_fcn)(x,xymax[1],usr) : au[j+1][i];
                        u_far = (j-1 <= 0)          ? (*usr->b_fcn)(x,xymin[1],usr) : au[j-1][i];
                    } else {
                        u_up  = (j+1 == info->my-1) ? (*usr->b_fcn)(x,xymax[1],usr) : au[j+1][i];
                        u_dn  = (j == 0)            ? (*usr->b_fcn)(x,xymin[1],usr) : au[j][i];
                        u_far = (j+2 >= info->my-1) ? (*usr->b_fcn)(x,xymax[1],usr) : au[j+2][i];
                    }
                // first-order upwind flux plus correction if have limiter
                flux = ap * u_up;
                if (limiter != NULL && u_dn != u_up) {
                    theta = (u_up - u_far) / (u_dn - u_up);
                    flux += ap * (*limiter)(theta) * (u_dn - u_up);
                }
                // update non-boundary and owned residual F_ij on both sides of computed flux
                // note: 1) aF[] does not have stencil width, 2) F_ij is scaled by scF = hx * hy
                iowned = (i >= info->xs);
                jowned = (j >= info->ys);
                ip1owned = (i+1 < info->xs + info->xm);
                jp1owned = (j+1 < info->ys + info->ym);
                switch (p) {
                    case 0:  // flux at E
                        if (i > 0 && j > 0 && iowned && jowned)
                            aF[j][i] += hy * flux;  // flux out of i,j at E
                        if (i+1 < info->mx-1 && j > 0 && ip1owned && jowned)
                            aF[j][i+1] -= hy * flux;  // flux into i+1,j at W
                        break;
                    case 1:  // flux at N
                        if (j > 0 && i > 0 && iowned && jowned)
                            aF[j][i] += hx * flux;  // flux out of i,j at N
                        if (j+1 < info->my-1 && i > 0 && iowned && jp1owned)
                            aF[j+1][i] -= hx * flux;  // flux into i,j+1 at S
                        break;
                }
            }
        }
    }
    // ff = flops per flux evaluation
    ff = (limiter == NULL) ? 6.0 : 13.0;
    if (limiter == &vanleer)
        ff += 4.0;
    ierr = PetscLogFlops(ff*2.0*(1.0+info->xm)*(1.0+info->ym)); CHKERRQ(ierr);
    return 0;
}

