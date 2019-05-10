static char help[] =
"Solves a 1D advection plus diffusion problem using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -lay_.\n"
"Equation is  - eps u'' + (a(x) u)' = 0  with a(x)=1, on domain [-1,1]\n"
"with Dirichlet boundary conditions u(-1) = 1, u(1) = 0.  Advection\n"
"discretized by first-order upwinding, centered, or van Leer limiter scheme.\n\n";

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

typedef struct {
    double      eps;          // amount of diffusion; require: eps > 0
    double      (*limiter_fcn)(double);
} AdCtx;

static double u_exact(double x, AdCtx *usr) {
    return (1.0 - exp((x-1) / usr->eps)) / (1.0 - exp(- 2.0 / usr->eps));
}

static double wind_a(double x) {
    return 1.0;
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double*,
                                        double*, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         hx, err2, errinf;
    DMDALocalInfo  info;
    LimiterType    limiter = NONE;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 1.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"lay_",
               "layer1 (1D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","positive diffusion coefficient",
               "layer1.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "layer1.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];

    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,3, // default to hx=1 grid
                 1,(user.limiter_fcn == NULL) ? 1 : 2, // stencil width
                 NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetApplicationContext(snes,&user); CHKERRQ(ierr);
    //if (limiter != VANLEER) {
    //    ierr = SNESSetType(snes,SNESKSPONLY); CHKERRQ(ierr);
    //}
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
         "done on %d point grid; eps = %g, limiter = %s\n",
         info.mx,user.eps,LimiterTypes[limiter]); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
    ierr = FormUExact(&info,&user,u_exact); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
    ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&err2); CHKERRQ(ierr);
    hx = 2.0 / (info.mx - 1);
    err2 *= PetscSqrtReal(hx);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "numerical error:  |u-uexact|_inf = %.4e,  |u-uexact|_{2,h} = %.4e\n",
         errinf,err2); CHKERRQ(ierr);

    VecDestroy(&u_exact);  SNESDestroy(&snes);
    return PetscFinalize();
}

PetscErrorCode FormUExact(DMDALocalInfo *info, AdCtx *usr, Vec uex) {
    PetscErrorCode  ierr;
    int          i;
    double       hx, x, *auex;

    hx = 2.0 / (info->mx - 1);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = -1.0 + i * hx;
        auex[i] = u_exact(x,usr);
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}

// compute residuals:  F_i = - eps u'' + u'
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double *au,
                                 double *aF, AdCtx *usr) {
    int          i;
    double       hx, halfx, hx2, x, uE, uW, uxx, a, u_up, flux;
    double       u_dn, u_far, theta;
    PetscBool    allowfar;

    hx = 2.0 / (info->mx - 1);
    halfx = hx / 2.0;
    hx2 = hx * hx;
    // for each owned cell, non-advective part of residual at cell center
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            aF[i] = au[i] - 1.0;
        } else if (i == info->mx-1) {
            aF[i] = au[i] - 0.0;
        } else {
            uW = (i == 1)          ? 1.0 : au[i-1];
            uE = (i == info->mx-2) ? 0.0 : au[i+1];
            uxx = (uW - 2.0 * au[i] + uE) / hx2;
            aF[i] = - usr->eps * uxx;
        }
    }
    // for each E face of an owned cell, compute flux at the face center
    // and then add that to the correct residual
    // note -1 start to get W faces of owned cells living on ownership
    // boundaries
    for (i=info->xs-1; i<info->xs+info->xm; i++) {
        // if cell center is outside [-1,1], or on x=1 boundary, then no need
        // to compute a flux
        if ((i < 0) || (i == info->mx-1))
            continue;
        // traverse E cell face center points x_{i+1/2} for flux contributions
        // get pth component of wind and first-order upwind flux
        x = -1.0 + i * hx;
        a = wind_a(x + halfx);
        if (a >= 0.0) {
            if (i == 0) {
                u_up = 1.0;
            } else {
                u_up = au[i];
            }
        } else {
            if (i+1 == info->mx-1) {
                u_up = 0.0;
            } else {
                u_up = au[i+1];
            }
        }
        flux = a * u_up;
        // flux correction if have limiter and not near boundaries
        if (usr->limiter_fcn != NULL) {
            allowfar = (i > 0 && i+1 < info->mx-1);
            if (allowfar) {
                // compute flux correction from high-order formula with psi(theta)
                u_dn = (a >= 0.0) ? au[i+1] : au[i];
                if (u_dn != u_up) {
                    u_far = (a >= 0.0) ? au[i-1] : au[i+2];
                    theta = (u_up - u_far) / (u_dn - u_up);
                    flux += a * (*usr->limiter_fcn)(theta) * (u_dn - u_up);
                }
            }
        }
        // update non-boundary and owned F_i on both sides of computed flux
        if (i > 0)
            aF[i] += flux / hx;  // flux out of i at E
        if (i+1 < info->mx-1 && i+1 < info->xs + info->xm)  // note aF[] does not have stencil width
            aF[i+1] -= flux / hx;  // flux into i+1 at W
    }
    return 0;
}

