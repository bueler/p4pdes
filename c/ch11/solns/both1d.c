static char help[] =
"Solves a 1D advection plus diffusion problem using FD discretization\n"
"and a structured-grid (DMDA).  Option prefix -b1_.  Equation is\n"
"  - eps u'' + (a(x) u)' = 0\n"
"with a(x)=1, on domain [-1,1], and with Dirichlet boundary conditions\n"
"u(-1) = 1, u(1) = 0.  Default eps=0.01.  The diffusion discretized by\n"
"centered, as usual, but advection is by first-order upwinding, centered,\n"
"or van Leer limiter scheme.  An analytic Jacobian is implemented, except\n"
"for the van Leer limiter.  The limiters in the residual and Jacobian\n"
"evaluations are separately controllable.\n\n";

#include <petsc.h>

typedef enum {NONE, CENTERED, VANLEER} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer",
                                     "LimiterType", "", NULL};

static PetscReal centered(PetscReal theta) {
    return 0.5;
}

static PetscReal vanleer(PetscReal theta) {
    const PetscReal abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

typedef PetscReal (*LimiterFcn)(PetscReal);
static LimiterFcn limiterptr[] = {NULL, &centered, &vanleer};

typedef struct {
    PetscReal   eps;          // amount of diffusion; require: eps > 0
    PetscReal   (*limiter_fcn)(PetscReal),
                (*jac_limiter_fcn)(PetscReal);
} AdCtx;

static PetscReal u_exact(PetscReal x, AdCtx *usr) {
    return (1.0 - exp((x-1) / usr->eps)) / (1.0 - exp(- 2.0 / usr->eps));
}

static PetscReal wind_a(PetscReal x) {
    return 1.0;
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, PetscReal*,PetscReal*, AdCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, PetscReal*, Mat, Mat, AdCtx*);

int main(int argc,char **argv) {
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    PetscReal      hx, err2, errinf;
    DMDALocalInfo  info;
    LimiterType    limiter = NONE, jac_limiter;
    PetscBool      snesfdset, snesfdcolorset;
    AdCtx          user;

    PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

    user.eps = 0.01;
    PetscOptionsBegin(PETSC_COMM_WORLD,"b1_",
               "both1d (1D advection-diffusion solver) options","");
    PetscCall(PetscOptionsReal("-eps","positive diffusion coefficient",
               "both1d.c",user.eps,&(user.eps),NULL));
    PetscCall(PetscOptionsEnum("-limiter","flux-limiter type",
               "both1d.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL));
    jac_limiter = limiter;
    PetscCall(PetscOptionsEnum("-jac_limiter","flux-limiter type used in Jacobian evaluation",
               "both1d.c",LimiterTypes,
               (PetscEnum)jac_limiter,(PetscEnum*)&jac_limiter,NULL));
    PetscOptionsEnd();

    if (user.eps <= 0.0) {
        SETERRQ(PETSC_COMM_SELF,2,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];
    PetscCall(PetscOptionsHasName(NULL,NULL,"-snes_fd",&snesfdset));
    PetscCall(PetscOptionsHasName(NULL,NULL,"-snes_fd_color",&snesfdcolorset));
    if (snesfdset || snesfdcolorset) {
        user.jac_limiter_fcn = NULL;
        jac_limiter = 4;   // corresponds to empty string
    } else
        user.jac_limiter_fcn = limiterptr[jac_limiter];

    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,
                 3, // default to hx=1 grid
                 1,2, // d.o.f., stencil width
                 NULL,&da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0));
    PetscCall(DMSetApplicationContext(da,&user));

    PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
    PetscCall(SNESSetDM(snes,da));
    PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user));
    PetscCall(DMDASNESSetJacobianLocal(da,
            (DMDASNESJacobian)FormJacobianLocal,&user));
    PetscCall(SNESSetApplicationContext(snes,&user));
    PetscCall(SNESSetFromOptions(snes));

    PetscCall(DMGetGlobalVector(da,&u_initial));
    PetscCall(VecSet(u_initial,0.0));
    PetscCall(SNESSolve(snes,NULL,u_initial));
    PetscCall(DMRestoreGlobalVector(da,&u_initial));
    PetscCall(DMDestroy(&da));

    PetscCall(SNESGetSolution(snes,&u));
    PetscCall(SNESGetDM(snes,&da_after));
    PetscCall(DMDAGetLocalInfo(da_after,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
         "done on %d point grid (eps = %g, limiter = %s, jac_limiter = %s)\n",
         info.mx,user.eps,LimiterTypes[limiter],LimiterTypes[jac_limiter]));

    PetscCall(DMCreateGlobalVector(da_after,&u_exact));
    PetscCall(FormUExact(&info,&user,u_exact));
    PetscCall(VecAXPY(u,-1.0,u_exact));    // u <- u + (-1.0) u_exact
    PetscCall(VecNorm(u,NORM_INFINITY,&errinf));
    PetscCall(VecNorm(u,NORM_2,&err2));
    hx = 2.0 / (info.mx - 1);
    err2 *= PetscSqrtReal(hx);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
         "numerical error:  |u-uexact|_inf = %.4e,  |u-uexact|_2 = %.4e\n",
         errinf,err2));

    VecDestroy(&u_exact);  SNESDestroy(&snes);
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode FormUExact(DMDALocalInfo *info, AdCtx *usr, Vec uex) {
    PetscInt   i;
    PetscReal  hx, x, *auex;

    hx = 2.0 / (info->mx - 1);
    PetscCall(DMDAVecGetArray(info->da, uex, &auex));
    for (i=info->xs; i<info->xs+info->xm; i++) {
        x = -1.0 + i * hx;
        auex[i] = u_exact(x,usr);
    }
    PetscCall(DMDAVecRestoreArray(info->da, uex, &auex));
    return 0;
}

/* compute residuals:
     F_i = (- eps u'' + (a(x) u)') * hx   at interior points
     F_i = c (u - (b.c.))          at boundary points         */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal *au,
                                 PetscReal *aF, AdCtx *usr) {
    const PetscReal  eps = usr->eps,
                     hx = 2.0 / (info->mx - 1),
                     halfx = hx / 2.0,
                     hx2 = hx * hx,
                     scdiag = (2.0 * eps) / hx + 1.0;
    PetscReal        x, uE, uW, uxx, a, u_up, flux, u_dn, u_far, theta;
    PetscInt         i;

    // for each owned cell, non-advective part of residual at cell center
    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0) {
            aF[i] = scdiag * (au[i] - 1.0);
        } else if (i == info->mx-1) {
            aF[i] = scdiag * (au[i] - 0.0);
        } else {
            uW = (i == 1)          ? 1.0 : au[i-1];
            uE = (i == info->mx-2) ? 0.0 : au[i+1];
            uxx = (uW - 2.0 * au[i] + uE) / hx2;
            aF[i] = - eps * uxx * hx;
        }
    }
    // for each E face of an owned cell, compute flux at the face center
    //     and then add that to the correct residual
    // note -1 start to get W faces of owned cells living on ownership
    //     boundaries
    for (i=info->xs-1; i<info->xs+info->xm; i++) {
        // if cell center is outside [-1,1], or on x=1 boundary, then no need
        // to compute a flux
        if (i < 0 || i == info->mx-1)
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
        if (usr->limiter_fcn != NULL) {
            // flux correction from high-order formula with psi(theta)
            if (a >= 0)
                u_dn = (i+1 < info->mx-1) ? au[i+1] : 0.0;
            else
                u_dn = au[i];
            if (u_dn != u_up) {
                if (a >= 0)
                   u_far = (i-1 > 0) ? au[i-1] : 1.0;
                else
                   u_far = (i+2 < info->mx-1) ? au[i+2] : 0.0;
                theta = (u_up - u_far) / (u_dn - u_up);
                flux += a * (*(usr->limiter_fcn))(theta) * (u_dn - u_up);
            }
        }
        // update non-boundary and owned F_i on both sides of computed flux
        // note aF[] does not have stencil width
        if (i > 0 && i >= info->xs)
            aF[i] += flux;  // flux out of i at E
        if (i+1 < info->mx-1 && i+1 < info->xs + info->xm)
            aF[i+1] -= flux;  // flux into i+1 at W
    }
    return 0;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal *u,
                                 Mat J, Mat P, AdCtx *usr) {
    const PetscReal eps = usr->eps,
                    hx = 2.0 / (info->mx - 1),
                    halfx = hx / 2.0,
                    scdiag = (2.0 * eps) / hx + 1.0;
    PetscInt        i, col[3];
    PetscReal       x, aE, aW, v[3];

    if (usr->jac_limiter_fcn == &vanleer) {
        SETERRQ(PETSC_COMM_SELF,1,"Jacobian for vanleer limiter is not implemented");
    }

    PetscCall(MatZeroEntries(P));

    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0 || i == info->mx-1) {
            v[0] = scdiag;
            PetscCall(MatSetValues(P,1,&i,1,&i,v,ADD_VALUES));
        } else {
            // diffusive part
            col[0] = i;
            v[0] = (2.0 * eps) / hx;
            col[1] = i-1;
            v[1] = (i-1 > 0) ? - eps / hx : 0.0;
            col[2] = i+1;
            v[2] = (i+1 < info->mx-1) ? - eps / hx : 0.0;
            PetscCall(MatSetValues(P,1,&i,3,col,v,ADD_VALUES));
            // advective part: from each adjacent face
            x = -1.0 + i * hx;
            aE = wind_a(x + halfx);
            aW = wind_a(x - halfx);
            if (aE >= 0.0) {
                col[0] = i;
                v[0] = aE;
            } else {
                col[0] = i+1;
                v[0] = (i+1 < info->mx-1) ? aE : 0.0;  // check if i+1 is boundary
            }
            if (aW >= 0.0) {
                col[1] = i-1;
                v[1] = (i-1 > 0) ? - aW : 0.0;  // check if i-1 is boundary
            } else {
                col[1] = i;
                v[1] = - aW;
            }
            PetscCall(MatSetValues(P,1,&i,2,col,v,ADD_VALUES));
            if (usr->jac_limiter_fcn == &centered) {
                col[0] = i+1;
                col[1] = i;
                if (aE >= 0.0) {
                    v[0] = (i+1 < info->mx-1) ? aE/2.0 : 0.0;  // check if i+1 is boundary
                    v[1] = - aE/2.0;
                } else {
                    v[0] = (i+1 < info->mx-1) ? - aE/2.0 : 0.0;  // check if i+1 is boundary
                    v[1] = aE/2.0;
                }
                PetscCall(MatSetValues(P,1,&i,2,col,v,ADD_VALUES));
                col[0] = i;
                col[1] = i-1;
                if (aW >= 0.0) {
                    v[0] = - aW/2.0;
                    v[1] = (i-1 > 0) ? aW/2.0 : 0.0;  // check if i-1 is boundary
                } else {
                    v[0] = aW/2.0;
                    v[1] = (i-1 > 0) ? - aW/2.0 : 0.0;  // check if i-1 is boundary
                }
                PetscCall(MatSetValues(P,1,&i,2,col,v,ADD_VALUES));
            }
        }
    }
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
    if (J != P) {
        PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
    }
    return 0;
}
