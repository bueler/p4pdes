static char help[] =
"Solves 2D advection plus diffusion problems using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -bth_.\n"
"Equation:\n"
"    - eps Laplacian u + div (a(x,y) u) = f(x,y),\n"
"where the (vector) wind a(x,y) and (scalar) source f(x,y) are given smooth\n"
"functions.  The domain is S = [-1,1]^2 with Dirichlet boundary conditions:\n"
"    u = g(x,y) on boundary S\n"
"where g(x,y) is a given smooth function.  Problems include: NOWIND, LAYER,\n"
"INTERNAL, and GLAZE.  The first of these has a=0 while the last three are\n"
"Examples 6.1.1, 6.1.3, and 6.1.4 in Elman et al (2014), respectively.\n"
"Advection can be discretized by first-order upwinding, centered, or a\n"
"van Leer limiter scheme.\n\n";

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
    double      eps;          // amount of diffusion; require: eps > 0
    double      (*limiter_fcn)(double),
                (*f_fcn)(double, double, void*),  // source
                (*g_fcn)(double, double, void*);  // boundary condition
} AdCtx;

// problem NOWIND: same basic problem as ./fish -fsh_problem manuexp
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

static double layer_f(double x, double y, void *user) {
    return 0.0;
}

static double layer_g(double x, double y, void *user) {
    return layer_u(x,y,user);
}

static double small = 1.0e-10;

// problem INTERNAL:  Elman page 239, Example 6.1.3
static double internal_f(double x, double y, void *user) {
    return 0.0;
}

static double internal_g(double x, double y, void *user) {
FIXME    if (x > 1.0-small) && ((y > -1.0+small) || (y < 1.0-small))) {
       return 1.0;
    else
       return 0.0;
}

// problem GLAZE:  Elman page 240, Example 6.1.4
static double glaze_f(double x, double y, void *user) {
    return 0.0;  // note wind is divergence-free
}

static double glaze_g(double x, double y, void *user) {
    if (x > 1.0-small) && ((y > -1.0+small) || (y < 1.0-small))) {
       return 1.0;
    else
       return 0.0;
}

static void* fptr[] = {&nowind_f, &layer_f, &internal_f, &glaze_f};
static void* gptr[] = {&nowind_g, &layer_g, &internal_g, &glaze_g};

FIXME

/* This vector function returns q=0,1,2 component.  It is used in
FormFunctionLocal() to get a(x,y,z). */
static double wind_a(double x, double y, double z, int q, AdCtx *user) {
    if (user->problem == LAYER) {
        return (q == 0) ? 1.0 : 0.0;
    } else if (user->problem == NOWIND) {
        return 0.0;
    } else { // GLAZE
        switch (q) {
            case 0:
                return 2.0 * z * (1.0 - x * x);
                break;
            case 1:
                return user->glaze_drift;
                break;
            case 2:
                return - 2.0 * x * (1.0 - z * z);
                break;
            default:
                return 1.0e308 * 100.0;  // cause overflow
        }
    }
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double***,
                                        double***, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         hx, hy, hz, err;
    int            my;
    char           filename[PETSC_MAX_PATH_LEN] = "filename.vtr";
    PetscBool      vtkoutput = PETSC_FALSE;
    PetscViewer    viewer;
    DMDALocalInfo  info;
    LimiterType    limiter = CENTERED;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 1.0;
    user.glaze_drift = 0.0;
    user.problem = LAYER;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ad3_",
               "ad3 (3D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","positive diffusion coefficient",
               "ad3.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-glaze_drift","y-direction drift constant for glaze problem",
               "ad3.c",user.glaze_drift,&(user.glaze_drift),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "ad3.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsString("-o","output solution in VTK format (.vtr,.vts), e.g. for paraview",
               "ad3.c",filename,filename,sizeof(filename),&vtkoutput);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem","problem type",
               "ad3.c",ProblemTypes,
               (PetscEnum)(user.problem),(PetscEnum*)&(user.problem),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }

    user.limiter_fcn = limiterptr[limiter];
    user.g_fcn = gptr[user.problem];
    user.b_fcn = bptr[user.problem];

    my = (user.limiter_fcn == NULL) ? 6 : 5;
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR,               // no diagonal differencing
        6,my,6,                          // usually default to hx=hx=hz=0.4 grid
                                         // (mz>=5 necessary for -snes_fd_color)
        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
        1,                               // d.o.f
        (user.limiter_fcn == NULL) ? 1 : 2, // stencil width
        NULL,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,0,""); CHKERRQ(ierr);

    // set coordinates of cell-centered regular grid
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    hy = 2.0 / info.my;
    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,
                                        -1.0+hy/2.0,1.0-hy/2.0,
                                        -1.0,1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(snes,&user); CHKERRQ(ierr);
    if (limiter != VANLEER) {
        ierr = SNESSetType(snes,SNESKSPONLY); CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMGetGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    hx = 2.0 / (info.mx - 1);
    hy = 2.0 / info.my;
    hz = 2.0 / (info.mz - 1);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on problem = %s, eps = %g, limiter = %s\n",
         ProblemTypes[user.problem],user.eps,LimiterTypes[limiter]); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "grid:  %d x %d x %d,  cell dims: %.4f x %.4f x %.4f\n",
         info.mx,info.my,info.mz,hx,hy,hz); CHKERRQ(ierr);

    if (vtkoutput) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,
            "writing solution_u to %s ...\n",filename); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)u, "solution_u"); CHKERRQ(ierr);
        ierr = PetscViewerVTKOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
        ierr = VecView(u,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }

    if ((user.problem == LAYER) || (user.problem == NOWIND)) {
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormUExact(&info,&user,u_exact); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
        ierr = VecNorm(u,NORM_2,&err); CHKERRQ(ierr);
        err *= PetscSqrtReal(hx * hy * hz);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
             "numerical error:  |u-uexact|_{2,h} = %.4e\n",err); CHKERRQ(ierr);
        ierr = VecDestroy(&u_exact); CHKERRQ(ierr);
    }

    ierr = SNESDestroy(&snes); CHKERRQ(ierr);
    return PetscFinalize();
}

PetscErrorCode FormUExact(DMDALocalInfo *info, AdCtx *usr, Vec uex) {
    PetscErrorCode  ierr;
    int          i, j, k;
    double       hx, hy, hz, x, y, z, ***auex;

    if ((usr->problem != LAYER) && (usr->problem != NOWIND)) {
        SETERRQ(PETSC_COMM_WORLD,1,"exact solutions only available for LAYER and NOWIND");
    }
    hx = 2.0 / (info->mx - 1);
    hy = 2.0 / info->my;
    hz = 2.0 / (info->mz - 1);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + (j + 0.5) * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * hx;
                if (usr->problem == LAYER)
                    auex[k][j][i] = layer_u(x,y,z,usr);
                else if (usr->problem == NOWIND)
                    auex[k][j][i] = nowind_u(x,y,z,usr);
                else {
                    SETERRQ(PETSC_COMM_WORLD,2,"how get here?");
                }
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}

/* compute residuals
    F_ijk = - eps Laplacian u - g(x,y,z) + div f
where the vector flux is
    f = a(x,y,z) u
*/
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

