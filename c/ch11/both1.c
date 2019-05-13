static char help[] =
"Solves a 1D advection plus diffusion problem using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -b1_.\n"
"Equation is  - eps u'' + (a(x) u)' = 0  with a(x)=1, on domain [-1,1]\n"
"with Dirichlet boundary conditions u(-1) = 1, u(1) = 0 and default eps=0.01.\n"
"Diffusion discretize by centered.  Advection discretized by first-order\n"
"upwinding, centered, or van Leer limiter scheme.\n\n";

/* fit error norms with two lines (and compare none,centered):
for LEV in 1 2 3 4 5 6 7 8 9 10 11 12 13; do ./both1 -da_refine $LEV -ksp_type preonly -pc_type lu -b1_limiter vanleer -snes_fd_color; done
*/

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
    double      (*limiter_fcn)(double),
                (*jac_limiter_fcn)(double);
} AdCtx;

static double u_exact(double x, AdCtx *usr) {
    return (1.0 - exp((x-1) / usr->eps)) / (1.0 - exp(- 2.0 / usr->eps));
}

static double wind_a(double x) {
    return 1.0;
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double*,double*, AdCtx*);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo*, double*, Mat, Mat, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         hx, err2, errinf;
    DMDALocalInfo  info;
    LimiterType    limiter = NONE, jac_limiter;
    PetscBool      snesfdset, snesfdcolorset;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 0.01;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"b1_",
               "both1 (1D advection and diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","positive diffusion coefficient",
               "both1.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "both1.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    jac_limiter = limiter;
    ierr = PetscOptionsEnum("-jac_limiter","flux-limiter type used in Jacobian evaluation",
               "both1.c",LimiterTypes,
               (PetscEnum)jac_limiter,(PetscEnum*)&jac_limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    user.limiter_fcn = limiterptr[limiter];

    ierr = PetscOptionsHasName(NULL,NULL,"-snes_fd",&snesfdset); CHKERRQ(ierr);
    ierr = PetscOptionsHasName(NULL,NULL,"-snes_fd_color",&snesfdcolorset); CHKERRQ(ierr);
    if (snesfdset || snesfdcolorset)
        user.jac_limiter_fcn = NULL;
    else {
        if (jac_limiter != NONE) {  //FIXME
            SETERRQ(PETSC_COMM_WORLD,99,"jac_limiter != NONE NOT IMPLEMENTED");
        }
        if (user.limiter_fcn == NULL && jac_limiter != NONE) {
            SETERRQ(PETSC_COMM_WORLD,1,"if limiter=none then jac_limiter=none is required");
        }
        user.jac_limiter_fcn = limiterptr[jac_limiter];
    }

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,2,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }

    ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,
                 3, // default to hx=1 grid
                 1, // d.o.f.
                 (user.limiter_fcn == NULL) ? 1 : 2, // stencil width
                 NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0,1.0); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
            (DMDASNESJacobian)FormJacobianLocal,&user); CHKERRQ(ierr);
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
         "numerical error:  |u-uexact|_inf = %.4e,  |u-uexact|_2 = %.4e\n",
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

// compute residuals with symmetric dependence:
//   F_i = (- eps u'' + u') * hx   at interior points
//   F_i = c (u - (b.c.))          at boundary points
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double *au,
                                 double *aF, AdCtx *usr) {
    int          i;
    const double eps = usr->eps,
                 hx = 2.0 / (info->mx - 1),
                 halfx = hx / 2.0,
                 hx2 = hx * hx,
                 scdiag = (2.0 * eps) / hx + 1.0;
    double       x, uE, uW, uxx, a, u_up, flux, u_dn, u_far, theta;
    PetscBool    allowfar;

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
    // and then add that to the correct residual
    // note -1 start to get W faces of owned cells living on ownership
    // boundaries
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
            u_dn = (a >= 0.0) ? au[i+1] : au[i];
            if (u_dn != u_up) {
                if (a >= 0)
                   u_far = (i-1 > 0) ? au[i-1] : 1.0;
                else
                   u_far = (i+2 < info->mx-1) ? au[i+2] : 0.0;
                theta = (u_up - u_far) / (u_dn - u_up);
                flux += a * (*usr->limiter_fcn)(theta) * (u_dn - u_up);
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

//FIXME initial implementation is for limiter=none (first-order upwind)
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, double *u,
                                 Mat J, Mat P, AdCtx *usr) {
    PetscErrorCode ierr;
    const double eps = usr->eps,
                 hx = 2.0 / (info->mx - 1),
                 halfx = hx / 2.0,
                 scdiag = (2.0 * eps) / hx + 1.0;
    int          i, col[3];
    double       a, x, v[3];

    ierr = MatZeroEntries(P); CHKERRQ(ierr);

    for (i=info->xs; i<info->xs+info->xm; i++) {
        if (i == 0 || i == info->mx-1) {
            v[0] = scdiag;
            ierr = MatSetValues(P,1,&i,1,&i,v,ADD_VALUES); CHKERRQ(ierr);
        } else {
            // diffusive part
            col[0] = i;
            v[0] = (2.0 * eps) / hx;
            col[1] = i-1;
            v[1] = (i-1 > 0) ? - eps / hx : 0.0;
            col[2] = i+1;
            v[2] = (i+1 < info->mx-1) ? - eps / hx : 0.0;
            ierr = MatSetValues(P,1,&i,3,col,v,ADD_VALUES); CHKERRQ(ierr);
            // advective part: from each adjacent face
            x = -1.0 + i * hx;
            a = wind_a(x + halfx);  // E face
            if (a >= 0.0) {
                col[0] = i;
                v[0] = a;
            } else {
                col[0] = i+1;
                v[0] = (i+1 < info->mx-1) ? a : 0.0;  // check if i+1 is boundary
            }
            a = wind_a(x - halfx);  // W face
            if (a >= 0.0) {
                col[1] = i-1;
                v[1] = (i-1 > 0) ? - a : 0.0;  // check if i-1 is boundary
            } else {
                col[1] = i;
                v[1] = - a;
            }
            ierr = MatSetValues(P,1,&i,2,col,v,ADD_VALUES); CHKERRQ(ierr);
        }
    }
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != P) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    return 0;
}

