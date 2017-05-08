static char help[] =
"Solves a 3D linear advection-diffusion problem with structured-grid (DMDA)\n"
"and SNES.  Option prefix -ad3_.  The equation is\n"
"    div (a(x,y,z) u) = eps Laplacian u + g(x,y,z,u),\n"
"where eps > 0, the vector velocity a(x,y,z) and the scalar source g(x,y,z,u)\n"
"are smooth.  The domain is  [-1,1]^3  with boundary conditions\n"
"    u(1,y,z) = b(y,z)\n"
"    u(-1,y,z) = u(x,-1,z) = u(x,1,z) = 0\n"
"    u periodic in z\n"
"The SNES sees as residual function the discretized version of\n"
"    F(u) = div (a(x,y,z) u) - eps Laplacian u - g(x,y,z,u)\n"
"so that Jacobian is positive-(semi-)definite if a=0 and g=0.\n"
"Significant restrictions are:\n"
"    * only Dirichlet and periodic boundary conditions are demonstrated\n"
"    * a(x,y,z), g(x,y,z,u), b(y,z) must be given by formulas\n"
"    * FIXME  only centered and first-order-upwind differences for advection\n"
"An exact solution is used to evaluate numerical error:\n"
"    u(x,y,z) = U(x) sin(E (y+1)) sin(F (z+1))\n"
"where  U(x) = (exp((x+1)/eps) - 1) / (exp(2/eps) - 1)\n"
"and constants E,F so that homogeneous/periodic boundary conditions\n"
"are satisfied.  The problem solved has  a=<1,0,0>,  b(y,z) = u(1,y,z),\n"
"and g(x,y,z,u) = eps lambda^2 u  where  lambda^2 = E^2 + F^2.\n\n";

/* evidence for convergence plus some feedback on iterations, but bad KSP iterations because GMRES+BJACOBI+ILU:
  $ for LEV in 0 1 2 3 4 5 6; do timer mpiexec -n 4 ./ad3 -snes_monitor -snes_converged_reason -ksp_converged_reason -ksp_rtol 1.0e-14 -da_refine $LEV; done

can go to LEV 7 if -ksp_type bicg or -ksp_type bcgs  ... GMRES is mem hog

eventually untuned algebraic multigrid is superior (tip-over point at -da_refine 6):
$ timer ./ad3 -snes_monitor -ksp_converged_reason -ksp_rtol 1.0e-11 -da_refine 6 -pc_type gamg
$ timer ./ad3 -snes_monitor -ksp_converged_reason -ksp_rtol 1.0e-11 -da_refine 6

all of these work:
  ./ad3 -snes_monitor -ksp_type preonly -pc_type lu
  "                   -snes_fd
  "                   -snes_mf
  "                   -snes_mf_operator

excellent illustration of Elman's point that discretization must handle advection problem if we are to get good preconditioning; compare following with and without -ad3_limiter none:
for LEV in 0 1 2 3 4 5; do timer mpiexec -n 4 ./ad3 -snes_converged_reason -ksp_converged_reason -ksp_rtol 1.0e-14 -da_refine $LEV -pc_type gamg -ad3_eps 0.1 -ad3_limiter none; done

FIXME: geometric multigrid?

FIXME: double glazing problem?
*/

#include <petsc.h>


/* compare limiters in advect.c */
static double centered(double theta) {
    return 0.5;
}

static double vanleer(double theta) {
    const double abstheta = PetscAbsReal(theta);
    return 0.5 * (theta + abstheta) / (1.0 + abstheta);
}

typedef enum {NONE, CENTERED, VANLEER} LimiterType;
static const char *LimiterTypes[] = {"none","centered","vanleer",
                                     "LimiterType", "", NULL};
static void* limiterptr[] = {NULL, &centered, &vanleer};


typedef struct {
    double      eps;
    LimiterType limiter;
    double      (*limiter_fcn)(double);
} Ctx;

static double a_wind(double x, double y, double z, int q, Ctx *user) {
    return (q == 0) ? 1.0 : 0.0;
}

static double EE = PETSC_PI / 2.0,
              FF = 2.0 * PETSC_PI;

static double u_exact(double x, double y, double z, Ctx *user) {
    const double C = exp(-2.0 / user->eps); // may underflow to 0; thats o.k.
    double u;
    u = (exp((x-1) / user->eps) - C) / (1.0 - C);
    return u * sin(EE*(y+1.0)) * sin(FF*(z+1.0));
}

static double g_source(double x, double y, double z, double u, Ctx *user) {
    const double lam2 = EE*EE + FF*FF; // lambda = sqrt(17.0) * PETSC_PI / 2.0
    return user->eps * lam2 * u_exact(x,y,z,user);
}

static double dgdu_source(double x, double y, double z, double u, Ctx *user) {
    return 0.0;
}

static double b_bdry(double y, double z, Ctx *user) {
    return sin(EE*(y+1.0)) * sin(FF*(z+1.0));
}


typedef struct {
    double  hx, hy, hz, halfx, halfy, halfz, hx2, hy2, hz2;
} Spacings;

void getSpacings(DMDALocalInfo *info, Spacings *s) {
    s->hx = 2.0 / (info->mx - 1);
    s->hy = 2.0 / (info->my - 1);
    s->hz = 2.0 / info->mz;    // periodic direction
    s->halfx = s->hx / 2.0;
    s->halfy = s->hy / 2.0;
    s->halfz = s->hz / 2.0;
    s->hx2 = s->hx * s->hx;
    s->hy2 = s->hy * s->hy;
    s->hz2 = s->hz * s->hz;
}


PetscErrorCode configureCtx(Ctx *usr) {
    PetscErrorCode  ierr;
    usr->eps = 1.0;
    usr->limiter = CENTERED;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ad3_",
               "ad3 (3D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","diffusion coefficient eps with  0 < eps < infty",
               "ad3.c",usr->eps,&(usr->eps),NULL); CHKERRQ(ierr);
    if (usr->eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",usr->eps);
    }
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "ad3.c",LimiterTypes,
           (PetscEnum)usr->limiter,(PetscEnum*)&usr->limiter,NULL); CHKERRQ(ierr);
    usr->limiter_fcn = limiterptr[usr->limiter];
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    return 0;
}


PetscErrorCode formUex(DMDALocalInfo *info, Ctx *usr, Vec uex) {
    PetscErrorCode  ierr;
    int          i, j, k;
    Spacings     s;
    double       x, y, z, ***auex;

    getSpacings(info,&s);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + (k+0.5) * s.hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + j * s.hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * s.hx;
                auex[k][j][i] = u_exact(x,y,z,usr);
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}


PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double ***au,
                                 double ***aF, Ctx *usr) {
    int          i, j, k, q, di, dj, dk;
    double       x, y, z, uu, uE, uW, uN, uS, uxx, uyy, uzz,
                 a, flux, u_up, u_dn, u_far, theta;
    PetscBool    deep;
    Spacings     s;

    getSpacings(info,&s);

    // clear F first
    for (k=info->zs; k<info->zs+info->zm; k++)
        for (j=info->ys; j<info->ys+info->ym; j++)
            for (i=info->xs; i<info->xs+info->xm; i++)
                aF[k][j][i] = 0.0;

    for (k=info->zs-1; k<info->zs+info->zm; k++) { // note -1 start
        z = -1.0 + (k+0.5) * s.hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + j * s.hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * s.hx;
                // for cell centers, determine non-advective parts of residual
                // FIXME: multiply through by cell volume to get better scaling?
                if (k >= info->zs) {
                    if (i == info->mx-1) {
                        aF[k][j][i] = au[k][j][i] - b_bdry(y,z,usr);
                    } else if (i == 0 || j == 0 || j == info->my-1) {
                        aF[k][j][i] = au[k][j][i];
                    } else {
                        uu = au[k][j][i];
                        uE = (i == info->mx-2) ? b_bdry(y,z,usr) : au[k][j][i+1];
                        uW = (i == 1)          ?             0.0 : au[k][j][i-1];
                        uN = (j == info->my-2) ?             0.0 : au[k][j+1][i];
                        uS = (j == 1)          ?             0.0 : au[k][j-1][i];
                        uxx = (uW - 2.0 * uu + uE) / s.hx2;
                        uyy = (uS - 2.0 * uu + uN) / s.hy2;
                        uzz = (au[k-1][j][i] - 2.0 * uu + au[k+1][j][i]) / s.hz2;
                        aF[k][j][i] -= usr->eps * (uxx + uyy + uzz)
                                       + g_source(x,y,z,uu,usr);
                    }
                }
                if (i == info->mx-1 || j == info->my-1)
                    continue;
                // traverse flux contributions on cell boundaries at E, N, T
                // [East/West, North/South, Top/Bottom for x,y,z resp.]
                for (q = 0; q < 3; q++) {
                    if (q < 2 && k < info->zs)  continue;
                    di = (q == 0) ? 1 : 0;
                    dj = (q == 1) ? 1 : 0;
                    dk = (q == 2) ? 1 : 0;
                    a = a_wind(x+s.halfx*di,y+s.halfy*dj,z+s.halfz*dk,q,usr);
                    u_up = (a >= 0.0) ? au[k][j][i] : au[k+dk][j+dj][i+di];
                    flux = a * u_up;
                    deep = (i > 1 && i < info->mx-2 && j > 1 && j < info->my-2);
                    if (usr->limiter_fcn != NULL && deep) {
                        u_dn = (a >= 0.0) ? au[k+dk][j+dj][i+di] : au[k][j][i];
                        if (u_dn != u_up) {
                            u_far = (a >= 0.0) ? au[k-dk][j-dj][i-di]
                                               : au[k+2*dk][j+2*dj][i+2*di];
                            theta = (u_up - u_far) / (u_dn - u_up);
                            flux += a * (*usr->limiter_fcn)(theta)*(u_dn-u_up);
                        }
                    }
                    // update non-boundary and owned F_ijk on both sides of computed flux
                    switch (q) {
                        case 0:  // flux at E
                            if (i > 0)
                                aF[k][j][i]   += flux / s.hx;
                            if (i < info->mx-1 && i+1 < info->xs + info->xm)
                                aF[k][j][i+1] -= flux / s.hx;
                            break;
                        case 1:  // flux at N
                            if (j > 0)
                                aF[k][j][i]   += flux / s.hy;
                            if (j < info->my-1 && j+1 < info->ys + info->ym)
                                aF[k][j+1][i] -= flux / s.hy;
                            break;
                        case 3:  // flux at T
                            if (k >= info->zs)
                                aF[k][j][i]   += flux / s.hz;
                            if (k+1 < info->zs + info->zm)
                                aF[k+1][j][i] -= flux / s.hz;
                            break;
                    }
                }
            }
        }
    }
    return 0;
}


// major FIXME:  Jacobian uses old form of advection
typedef struct {
    double  x,y,z;
} Wind;

static Wind a_wind_old(double x, double y, double z) {
    Wind W = {1.0,0.0,0.0};
    return W;
}

PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar ***au,
                                 Mat J, Mat Jpre, Ctx *usr) {
    PetscErrorCode  ierr;
    int          i,j,k,q;
    double       v[7],diag,x,y,z;
    MatStencil   col[7],row;
    Spacings     s;
    Wind         W;

    getSpacings(info,&s);
    diag = usr->eps * 2.0 * (1.0/s.hx2 + 1.0/s.hy2 + 1.0/s.hz2);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * s.hz;
        row.k = k;
        col[0].k = k;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + j * s.hy;
            row.j = j;
            col[0].j = j;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * s.hx;
                row.i = i;
                col[0].i = i;
                if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
                    v[0] = 1.0;
                    q = 1;
                } else {
                    W = a_wind_old(x,y,z);
                    v[0] = diag - dgdu_source(x,y,z,au[k][j][i],usr);
                    if (!usr->limiter) {
                        v[0] += (W.x / s.hx) * ((W.x > 0.0) ? 1.0 : -1.0);
                        v[0] += (W.y / s.hy) * ((W.y > 0.0) ? 1.0 : -1.0);
                        v[0] += (W.z / s.hz) * ((W.z > 0.0) ? 1.0 : -1.0);
                    }
                    v[1] = - usr->eps / s.hz2;
                    if (!usr->limiter) {
                        if (W.z > 0.0)
                            v[1] -= W.z / s.hz;
                    } else
                        v[1] -= W.z / (2.0 * s.hz);
                    col[1].k = k-1;  col[1].j = j;  col[1].i = i;
                    v[2] = - usr->eps / s.hz2;
                    if (!usr->limiter) {
                        if (W.z <= 0.0)
                            v[2] += W.z / s.hz;
                    } else
                        v[2] += W.z / (2.0 * s.hz);
                    col[2].k = k+1;  col[2].j = j;  col[2].i = i;
                    q = 3;
                    if (i-1 != 0) {
                        v[q] = - usr->eps / s.hx2;
                        if (!usr->limiter) {
                            if (W.x > 0.0)
                                v[q] -= W.x / s.hx;
                        } else
                            v[q] -= W.x / (2.0 * s.hx);
                        col[q].k = k;  col[q].j = j;  col[q].i = i-1;
                        q++;
                    }
                    if (i+1 != info->mx-1) {
                        v[q] = - usr->eps / s.hx2;
                        if (!usr->limiter) {
                            if (W.x <= 0.0)
                                v[q] += W.x / s.hx;
                        } else
                            v[q] += W.x / (2.0 * s.hx);
                        col[q].k = k;  col[q].j = j;  col[q].i = i+1;
                        q++;
                    }
                    if (j-1 != 0) {
                        v[q] = - usr->eps / s.hy2;
                        if (!usr->limiter) {
                            if (W.y > 0.0)
                                v[q] -= W.y / s.hy;
                        } else
                            v[q] -= W.y / (2.0 * s.hy);
                        col[q].k = k;  col[q].j = j-1;  col[q].i = i;
                        q++;
                    }
                    if (j+1 != info->my-1) {
                        v[q] = - usr->eps / s.hy2;
                        if (!usr->limiter) {
                            if (W.y <= 0.0)
                                v[q] += W.y / s.hy;
                        } else
                            v[q] += W.y / (2.0 * s.hy);
                        col[q].k = k;  col[q].j = j+1;  col[q].i = i;
                        q++;
                    }
                }
                ierr = MatSetValuesStencil(Jpre,1,&row,q,col,v,INSERT_VALUES); CHKERRQ(ierr);
            }
        }
    }
    ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    return 0;
}


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    double         err;
    DMDALocalInfo  info;
    Spacings       s;
    Ctx            user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    ierr = configureCtx(&user); CHKERRQ(ierr);

    ierr = DMDACreate3d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC,
        DMDA_STENCIL_STAR,               // no diagonal differencing
        3,3,5,                           // default to hx=hx=0.5,hz=0.2 grid
                                         // (mz=5 allows -snes_fd_color)
        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
        1, 2,                            // d.o.f & stencil width
        NULL,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    if ((info.mx < 2) || (info.my < 2) || (info.mz < 3)) {
        SETERRQ(PETSC_COMM_WORLD,1,"grid too coarse: require (mx,my,mz) >= (2,2,3)");
    }
    getSpacings(&info,&s);
    ierr = DMDASetUniformCoordinates(da,-1.0,1.0,-1.0,1.0,-1.0+s.halfz,1.0-s.halfz); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (DMDASNESFunction)FormFunctionLocal,&user);CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
            (DMDASNESJacobian)FormJacobianLocal,&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = VecSet(u,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    ierr = formUex(&info,&user,uexact); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
    ierr = VecNorm(u,NORM_2,&err); CHKERRQ(ierr);
    err *= PetscSqrtReal(s.hx * s.hy * s.hz);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d x %d grid with eps=%g:  error |u-uexact|_{2,h} = %.4e\n",
         info.mx,info.my,info.mz,user.eps,err); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

