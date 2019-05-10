static char help[] =
"Solves a 1D advection plus diffusion problem using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -lay_.\n"
"Equation:\n"
"    - eps u'' + u' = 0,\n"
"The domain is [-1,1] with Dirichlet boundary conditions:\n"
"    u(-1) = 1 and u(+1) = 0\n"
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

typedef struct {
    double      eps;          // amount of diffusion; require: eps > 0
    double      (*limiter_fcn)(double);
} AdCtx;

static double u_exact(double x) {
    return FIXME;
}

extern PetscErrorCode FormUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double*,
                                        double*, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         err2, errinf;
    PetscViewer    viewer;
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

    // set coordinates of cell-centered regular grid
    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,
                                        -1.0,1.0,-1.0,1.0); CHKERRQ(ierr);

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
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d point grid; eps = %g, limiter = %s\n",
         info.mx,ProblemTypes[user.problem],user.eps,LimiterTypes[limiter]); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
    ierr = FormUExact(&info,&user,u_exact); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
    ierr = VecNorm(u,NORM_INF,&errinf); CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&err2); CHKERRQ(ierr);
    err2 *= PetscSqrtReal(hx);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "numerical error:  |u-uexact|=%.4e,  |u-uexact|_{2,h} = %.4e\n",
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
    int          i, p, di;
    double       hx, halfx, hx2, x,
                 uu, uE, uW, uT, uB, uxx, uyy, uzz,
                 ap, flux, u_up, u_dn, u_far, theta;
    PetscBool    allowdeep;

    hx = 2.0 / (info->mx - 1);
    halfx = hx / 2.0;
    hx2 = hx * hx;

FIXME

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

