static char help[] =
"Solves 2D advection-diffusion problems using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -bth_.\n"
"Equation:\n"
"    - eps Laplacian u + Div (a(x,y) u) = g(x,y),\n"
"where the (vector) wind a(x,y) and (scalar) source g(x,y) are given smooth\n"
"functions.  The domain is S = (-1,1)^2 with Dirichlet boundary conditions:\n"
"    u = b(x,y) on boundary S\n"
"where b(x,y) is a given smooth function.  Problems include: NOWIND, LAYER,\n"
"INTERNAL, and GLAZE.  The first of these has a=0 while the last three are\n"
"Examples 6.1.1, 6.1.3, and 6.1.4 in Elman et al (2014), respectively.\n"
"Advection can be discretized by first-order upwinding (none), centered, or a\n"
"van Leer limiter scheme.  Option allows switching to none limiter on all grids\n"
"with mesh Peclet above one.\n\n";

/*
1. looks like O(h^2) and good multigrid for NOWIND:
for LEV in 1 2 3 4 5 6 7 8; do
    ./both -snes_type ksponly -ksp_converged_reason -bth_problem nowind -da_refine $LEV -ksp_rtol 1.0e-10 -pc_type mg
done

2. same problem and scaling as fish.c for NOWIND with eps=1.0:
./both -bth_problem nowind -bth_eps 1.0 -ksp_view_mat ::ascii_dense
../ch6/fish -ksp_view_mat ::ascii_dense
and
./both -bth_problem nowind -bth_eps 1.0 -da_refine 2 -ksp_monitor_short -snes_type ksponly -ksp_type cg
../ch6/fish -da_refine 2 -ksp_monitor_short -fsh_initial_gonboundary 0

3. convergence at O(h^2) and apparent optimal order for LAYER with GMRES+GMG with GS smoothing and CENTERED on fine grid but otherwise first-order upwinding:
for LEV in 5 6 7 8 9 10; do
    ./both -snes_type ksponly -bth_limiter centered -bth_none_on_peclet -bth_problem layer -da_refine $LEV -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type sor -mg_levels_pc_sor_forward
done

4. visualize GLAZE but on a 1025x1025 grid using GMRES+GMG with ILU smoothing:
./both -bth_eps 0.005 -bth_limiter none -bth_problem glaze -snes_converged_reason -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -snes_monitor_solution draw -draw_pause 1 -da_refine 9

5. evidence of optimality for GLAZE using GMRES+GMG with ILU smoothing and a 33x33 coarse grid:
for LEV in 5 6 7 8 9 10; do
    ./both -bth_eps 0.005 -bth_limiter centered -bth_none_on_peclet -bth_problem glaze -snes_type ksponly -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -da_refine $LEV -pc_mg_levels $(( $LEV - 3 ))
done

6. good solver using BCGS for low memory, BOX stencil and 1 sweep ILU smoothing for efficient smoother, and right PC (why so much better?):
for LEV in 5 6 7 8 9 10; do
    ./both -snes_type ksponly -ksp_type bcgs -ksp_pc_side right -ksp_converged_reason -bth_problem glaze -bth_eps 0.005 -bth_limiter centered -bth_none_on_peclet -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -mg_levels_ksp_max_it 1 -da_refine $LEV -bth_stencil_box
done

7. try -ksp_type bcgs for memory savings relative to GMRES
*/

#include <petsc.h>

typedef enum {NONE, CENTERED, VANLEER, THIRD} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer","third",
                                     "LimiterType", "", NULL};

static double centered(double theta) {
    return 0.5;
}

static double vanleer(double theta) {
    const double abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);   // 4 flops
}

static double third(double theta) {
    if (theta > 0.0)
        return (1.0/3.0) + (1.0/6.0) * theta;   // 2 flops
    else
        return 0.0;
}

static void* limiterptr[] = {NULL, &centered, &vanleer, &third};

typedef enum {NOWIND, LAYER, INTERNAL, GLAZE} ProblemType;
static const char *ProblemTypes[] = {"nowind", "layer", "internal", "glaze",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType problem;
    double      eps;                              // diffusion eps > 0
    double      (*limiter_fcn)(double),
                (*g_fcn)(double, double, void*),  // source
                (*b_fcn)(double, double, void*);  // boundary condition
    PetscBool   none_on_peclet;                   // use none limiter when
    double      a_scale;                          //    mesh Peclet exceeds 1
} AdCtx;

// used for source functions
static double zero(double x, double y, void *user) {
    return 0.0;
}

// problem NOWIND: same problem as ./fish -fsh_problem manuexp
static double nowind_u(double x, double y, void *user) {  // exact solution
    return - x * exp(y);
}

static double nowind_g(double x, double y, void *user) {
    AdCtx* usr = (AdCtx*)user;
    return usr->eps * x * exp(y);
}

static double nowind_b(double x, double y, void *user) {
    return nowind_u(x,y,user);
}

// problem LAYER:  Elman page 237, Example 6.1.1
static double layer_u(double x, double y, AdCtx *user) {  // exact solution
    AdCtx* usr = (AdCtx*)user;
    return x * (1.0 - exp((y-1) / usr->eps)) / (1.0 - exp(- 2.0 / usr->eps));
}

static double layer_b(double x, double y, void *user) {
    return layer_u(x,y,user);
}

// problem INTERNAL:  Elman page 239, Example 6.1.3
static double internal_b(double x, double y, void *user) {
    if (y > 2*x - 1)
       return 1.0;   // along x=1 and (y=-1 & 0 < x < 1) boundaries
    else
       return 0.0;
}

// problem GLAZE:  Elman page 240, Example 6.1.4
static double glaze_b(double x, double y, void *user) {
    if (x > 0.0 && y < x && y > -x)
       return 1.0;   // along x=1 boundary
    else
       return 0.0;
}

static void* uexptr[] = {&nowind_u, &layer_u, NULL,        NULL};
static void* gptr[]   = {&nowind_g, &zero,    &zero,       &zero};
static void* bptr[]   = {&nowind_b, &layer_b, &internal_b, &glaze_b};

/* This vector function returns q=0,1 component.  It is used in
   FormFunctionLocal() to get a(x,y). */
static double wind_a(double x, double y, int q, AdCtx *user) {
    switch (user->problem) {
        case NOWIND:
            return 0.0;
        case LAYER:
            return (q == 0) ? 0.0 : 1.0;
        case INTERNAL:
            return (q == 0) ? -0.5 : PetscSqrtReal(3.0)/2.0;
        case GLAZE:
            return (q == 0) ? 2.0*y*(1.0-x*x) : -2.0*x*(1.0-y*y);
        default:
            return NAN;
    }
}

extern PetscErrorCode FormUExact(DMDALocalInfo*,AdCtx*, 
                                 double (*)(double, double, void*),Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*,double**,double**,AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u;
    DMDALocalInfo  info;
    double         (*uexact_fcn)(double, double, void*);
    LimiterType    limiter = NONE;
    PetscBool      init_exact = PETSC_FALSE, stencil_box = PETSC_FALSE;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 0.01;
    user.none_on_peclet = PETSC_FALSE;
    user.problem = LAYER;
    user.a_scale = 1.0;   // this could be made dependent on problem
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
               "on coarse grids such that mesh peclet exceeds 1, switch to none limiter",
               "both.c",user.none_on_peclet,&(user.none_on_peclet),NULL);
               CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem","problem type",
               "both.c",ProblemTypes,
               (PetscEnum)(user.problem),(PetscEnum*)&(user.problem),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-stencil_box","use box stencil; improves performance of ILU smoothing",
               "both.c",stencil_box,&stencil_box,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];
    uexact_fcn = uexptr[user.problem];
    user.g_fcn = gptr[user.problem];
    user.b_fcn = bptr[user.problem];

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
        (stencil_box) ? DMDA_STENCIL_BOX : DMDA_STENCIL_STAR,
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
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d grid (problem = %s, eps = %g, limiter = %s)\n",
         info.mx,info.my,ProblemTypes[user.problem],user.eps,LimiterTypes[limiter]);
         CHKERRQ(ierr);

    if (uexact_fcn != NULL) {
        Vec     u_exact;
        double  xymin[2], xymax[2], hx, hy, err2;
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormUExact(&info,&user,uexact_fcn,u_exact); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
        ierr = VecNorm(u,NORM_2,&err2); CHKERRQ(ierr);
        ierr = DMDAGetBoundingBox(da_after,xymin,xymax); CHKERRQ(ierr);
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
                          double (*uexact)(double, double, void*), Vec uex) {
    PetscErrorCode  ierr;
    int          i, j;
    double       xymin[2], xymax[2], hx, hy, x, y, **auex;

    if (uexact == NULL) {
        SETERRQ(PETSC_COMM_WORLD,1,"exact solution not available");
    }
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
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
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **aF, AdCtx *usr) {
    PetscErrorCode ierr;
    int          i, j, p;
    double       xymin[2], xymax[2], hx, hy, hx2, hy2, scF, scBC,
                 x, y, uE, uW, uN, uS, uxx, uyy,
                 ap, flux, u_up, u_dn, u_far, theta;
    double       (*limiter)(double);
    PetscBool    iowned, jowned, ip1owned, jp1owned;
    PetscLogDouble ff;

    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    limiter = usr->limiter_fcn;
    if (usr->none_on_peclet && usr->a_scale * PetscMax(hx,hy) / usr->eps > 1.0) {
        limiter = NULL;
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
    else if (limiter == &third)
        ff += 2.0;
    ierr = PetscLogFlops(ff*2.0*(1.0+info->xm)*(1.0+info->ym)); CHKERRQ(ierr);
    return 0;
}

