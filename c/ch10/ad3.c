static char help[] =
"Solves a 3D linear advection-diffusion problem using FD discretization,\n"
"structured-grid (DMDA), and SNES.  Option prefix -ad3_.  The equation is\n"
"    - eps Laplacian u + div (w_0 a(x,y,z) u) = g(x,y,z),\n"
"where the wind a(x,y,z) and source g(x,y,z) are given smooth functions.\n"
"The diffusivity eps > 0 (-ad3_eps) and wind constant (-ad3_w0) can be chosen\n"
"by options.  The domain is  [-1,1]^3  with Dirichlet-periodic boundary\n"
"conditions\n"
"    u(1,y,z) = b(y,z)\n"
"    u(-1,y,z) = u(x,y,-1) = u(x,y,1) = 0\n"
"    u periodic in y\n"
"where b(y,z) is a given smooth function.  An exact solution, based on\n"
"a boundary layer of width eps, and a double-glazing problem are included\n"
"(-ad3_problem layer|glaze).  Advection can be discretized by first-order\n"
"upwinding, centered, or van Leer limiter schemes\n"
"(-ad3_limiter none|centered|vanleer).\n\n";

/* TODO:
1. transpose so both layer and glaze problems seem more natural
2. put coefficient on wind to allow turning off
3. check that Poisson equation in good shape including GMG
4. implement glaze problem
5. multiply through by cell volume to get better scaling ... this may be why GMG is not good
6. options to allow view slice
*/

/* OLD NOTES CONTAINING INTERESTING/RELEVANT IDEAS
evidence for convergence plus some feedback on iterations, but bad KSP iterations because GMRES+BJACOBI+ILU:
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

multigrid works, but is not beneficial yet; compare
    ./ad3 -{snes,ksp}_converged_reason -ad3_limiter none -snes_type ksponly -da_refine 4 -pc_type X
with X=ilu,mg,gamg;  presumably needs advection-specific smoothing

grid-sequencing works, in nonlinear limiter case, but not beneficial; compare
    ./ad3 -{snes,ksp}_converged_reason -ad3_limiter vanleer -da_refine 4
    ./ad3 -{snes,ksp}_converged_reason -ad3_limiter vanleer -snes_grid_sequence 4
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

typedef enum {LAYER, GLAZE} ProblemType;
static const char *ProblemTypes[] = {"layer", "glaze",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType problem;
    double      eps, w0, glazedrift;
    double      (*limiter_fcn)(double);
} AdCtx;

/* problem LAYER:
A partly-manufactured 3D exact solution of a boundary layer problem, with
exponential layer near x=1, allows evaluation of numerical error:
    u(x,y,z) = U(x) sin(E (y+1)) sin(F (z+1))
where
    U(x) = (exp((x+1)/delta) - 1) / (exp(2/delta) - 1)
         = (exp((x-1)/delta) - C) / (1 - C)
where  delta = eps / w0  and  C = exp(-2 / delta).  Note eps > 0 and w0 > 0
are required.  Note C may gracefully underflow.  Thus U(x) satisfies
    -eps U'' + w0 U' = 0.
Note U(x) satisfies U(-1)=0 and U(1)=1, and it has a boundary layer of width
delta near x=1.  Constants E = 2 pi and F = pi / 2 are set so that u is periodic
and smooth in y and satisfies Dirichlet boundary conditions u(x,y,+-1) = 0.
The problem solved has
    a = <1,0,0>
    g(x,y,z,u) = lambda u
where lambda = eps (E^2 + F^2), and
    b(y,z) = u(1,y,z)
*/

static double EE = 2.0 * PETSC_PI,
              FF = PETSC_PI / 2.0;

static double layer_u(double x, double y, double z, AdCtx *user) {
    const double delta = user->eps / user->w0,
                 C = exp(-2.0 / delta); // may underflow to 0; that's o.k.
    return ((exp((x-1) / delta) - C) / (1.0 - C))
           * sin(EE*(y+1.0)) * sin(FF*(z+1.0));
}

static double layer_g(double x, double y, double z, AdCtx *user) {
    const double lam = user->eps * (EE*EE + FF*FF);
    return lam * layer_u(x,y,z,user);
}

/* problem GLAZE:
See pages 240-241 of Elman et al (2014) for this problem.  The recirculating
flow is counterclockwise in the x-z plane.  An additional drift is in the
y-direction.
*/

// vector function returns q=0,1,2 component
static double wind_a(double x, double y, double z, int q, AdCtx *user) {
    if (user->problem == LAYER) {
        return (q == 0) ? 1.0 : 0.0;
    } else {
        switch (q) {
            case 0:
                return 2.0 * z * (1.0 - x * x);
                break;
            case 1:
                return user->glazedrift;
                break;
            case 2:
                return - 2.0 * x * (1.0 - z * z);
                break;
            default:
                return 9.99999e306;  // FIXME
        }
    }
}

static double source_g(double x, double y, double z, AdCtx *user) {
    if (user->problem == LAYER) {
        return layer_g(x,y,z,user);
    } else {
        return 0.0;
    }
}

static double bdry_b(double y, double z, AdCtx *user) {
    if (user->problem == LAYER) {
        return layer_u(1.0,y,z,user);
    } else {
        return 1.0;
    }
}

extern PetscErrorCode FormLayerUExact(DMDALocalInfo*, AdCtx*, Vec);
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo*, double***,
                                        double***, AdCtx*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da, da_after;
    SNES           snes;
    Vec            u_initial, u, u_exact;
    double         hx, hy, hz, err;
    int            my;
    DMDALocalInfo  info;
    LimiterType    limiter = CENTERED;
    AdCtx          user;

    PetscInitialize(&argc,&argv,(char*)0,help);

    user.eps = 1.0;
    user.w0 = 1.0;
    user.glazedrift = 0.0;  // FIXME option
    user.problem = LAYER;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"ad3_",
               "ad3 (3D advection-diffusion solver) options",""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps","diffusion coefficient eps with  0 < eps < infty",
               "ad3.c",user.eps,&(user.eps),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-limiter","flux-limiter type",
               "ad3.c",LimiterTypes,
               (PetscEnum)limiter,(PetscEnum*)&limiter,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem","problem type",
               "ad3.c",ProblemTypes,
               (PetscEnum)(user.problem),(PetscEnum*)&(user.problem),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-w0","overall scaling of wind (velocity)",
               "ad3.c",user.w0,&(user.w0),NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    if (user.eps <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,1,"eps=%.3f invalid ... eps > 0 required",user.eps);
    }
    if (user.w0 <= 0.0) {
        SETERRQ1(PETSC_COMM_WORLD,2,"w0=%.3f invalid ... w0 > 0 required",user.eps);
    }
    user.limiter_fcn = limiterptr[limiter];

    my = (user.limiter_fcn == NULL) ? 6 : 5;
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
        DM_BOUNDARY_NONE, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_NONE,
        DMDA_STENCIL_STAR,               // no diagonal differencing
        6,my,6,                          // usually default to hx=hx=hz=0.4 grid
                                         // (mz>=5 allows -snes_fd_color)
        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
        1,                               // d.o.f
        (user.limiter_fcn == NULL) ? 1 : 2, // stencil width
        NULL,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);
    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);

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
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u_initial); CHKERRQ(ierr);
    ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);
    ierr = SNESSolve(snes,NULL,u_initial); CHKERRQ(ierr);
    ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
    ierr = DMDestroy(&da); CHKERRQ(ierr);

    ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr);
    ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
    hx = 2.0 / (info.mx - 1);
    hy = 2.0 / info.my;
    hz = 2.0 / (info.mz - 1);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "done on %d x %d x %d grid, cell dims %.4f x %.4f x %.4f, eps=%g, limiter = %s",
         info.mx,info.my,info.mz,hx,hy,hz,user.eps,LimiterTypes[limiter]); CHKERRQ(ierr);

    if (user.problem == LAYER) {
        ierr = VecDuplicate(u,&u_exact); CHKERRQ(ierr);
        ierr = FormLayerUExact(&info,&user,u_exact); CHKERRQ(ierr);
        ierr = VecAXPY(u,-1.0,u_exact); CHKERRQ(ierr);    // u <- u + (-1.0) u_exact
        ierr = VecNorm(u,NORM_2,&err); CHKERRQ(ierr);
        err *= PetscSqrtReal(hx * hy * hz);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
             "\n  error |u-uexact|_{2,h} = %.4e\n",err); CHKERRQ(ierr);
        VecDestroy(&u_exact);
    } else {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"...\n"); CHKERRQ(ierr);
    }

    SNESDestroy(&snes);
    return PetscFinalize();
}

PetscErrorCode FormLayerUExact(DMDALocalInfo *info, AdCtx *usr, Vec uex) {
    PetscErrorCode  ierr;
    int          i, j, k;
    double       hx, hy, hz, x, y, z, ***auex;

    hx = 2.0 / (info->mx - 1);
    hz = 2.0 / info->my;
    hy = 2.0 / (info->mz - 1);
    ierr = DMDAVecGetArray(info->da, uex, &auex);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + j * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = -1.0 + (j + 0.5) * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * hx;
                auex[k][j][i] = layer_u(x,y,z,usr);
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, uex, &auex);CHKERRQ(ierr);
    return 0;
}

/* compute residuals
    F_ijk = - eps Laplacian u - g(x,y,z) + div f
where the vector flux is
    f = w_0 a(x,y,z) u
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double ***au,
                                 double ***aF, AdCtx *usr) {
    int          i, j, k, p, di, dj, dk;
    double       hx, hy, hz, halfx, halfy, halfz, hx2, hy2, hz2, x, y, z,
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

    // clear F first
    for (k=info->zs; k<info->zs+info->zm; k++)
        for (j=info->ys; j<info->ys+info->ym; j++)
            for (i=info->xs; i<info->xs+info->xm; i++)
                aF[k][j][i] = 0.0;

    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = -1.0 + k * hz;
        for (j=info->ys-1; j<info->ys+info->ym; j++) { // note -1 start
            y = -1.0 + (j + 0.5) * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = -1.0 + i * hx;
                // for owned cell centers, determine non-advective parts of residual
                // notation East/West, North/South, Top/Bottom for x,y,z resp.
                // FIXME: multiply through by cell volume to get better scaling?
                if (j >= info->ys) {
                    if (i == info->mx-1) {
                        aF[k][j][i] = au[k][j][i] - bdry_b(y,z,usr);
                    } else if (i == 0 || k == 0 || k == info->mz-1) {
                        aF[k][j][i] = au[k][j][i];
                    } else {
                        uu = au[k][j][i];
                        uE = (i == info->mx-2) ? bdry_b(y,z,usr) : au[k][j][i+1];
                        uW = (i == 1)          ?             0.0 : au[k][j][i-1];
                        uT = (k == info->mz-2) ?             0.0 : au[k+1][j][i];
                        uB = (k == 1)          ?             0.0 : au[k-1][j][i];
                        uxx = (uW - 2.0 * uu + uE) / hx2;
                        uyy = (au[k][j-1][i] - 2.0 * uu + au[k][j+1][i]) / hy2;
                        uzz = (uB - 2.0 * uu + uT) / hz2;
                        aF[k][j][i] -= usr->eps * (uxx + uyy + uzz)
                                       + source_g(x,y,z,usr);
                    }
                }
                // FIXME: is the following correct?  if we are on x=1 or z=1
                // boundaries then do we not need to compute any face-center fluxes?
                if (i == info->mx-1 || k == info->mz-1)
                    continue;
                // traverse half of cell face center points for flux contributions
                // E,N,T corresponding to p=0,1,2
                for (p = 0; p < 3; p++) {
                    if (j < info->ys && p != 1)  continue;
                    di = (p == 0) ? 1 : 0;
                    dj = (p == 1) ? 1 : 0;
                    dk = (p == 2) ? 1 : 0;
                    // get pth component of wind
                    ap = wind_a(x + halfx * di, y + halfy * dj, z + halfz * dk,
                                p,usr);
                    ap *= usr->w0;
                    u_up = (ap >= 0.0) ? au[k][j][i] : au[k+dk][j+dj][i+di];
                    flux = ap * u_up;  // first-order upwind flux
                    // flux correction is not possible too-near boundaries
                    allowdeep = (   i > 1 && i < info->mx-2
                                 && k > 1 && k < info->mz-2);
                    if (usr->limiter_fcn != NULL && allowdeep) {
                        u_dn = (ap >= 0.0) ? au[k+dk][j+dj][i+di] : au[k][j][i];
                        if (u_dn != u_up) {
                            u_far = (ap >= 0.0) ? au[k-dk][j-dj][i-di]
                                                : au[k+2*dk][j+2*dj][i+2*di];
                            theta = (u_up - u_far) / (u_dn - u_up);
                            flux += ap * (*usr->limiter_fcn)(theta) * (u_dn - u_up);
                        }
                    }
                    // update non-boundary and owned F_ijk on both sides of computed flux
                    switch (p) {
                        case 0:  // flux at E
                            if (i > 0)
                                aF[k][j][i]   += flux / hx;  // flux out of i,j,k at E
                            if (i < info->mx-1 && i+1 < info->xs + info->xm)
                                aF[k][j][i+1] -= flux / hx;  // flux into i+1,j,k at W
                            break;
                        case 1:  // flux at N
                            if (j >= info->ys)
                                aF[k][j][i]   += flux / hy;
                            if (j+1 < info->ys + info->ym)
                                aF[k][j+1][i] -= flux / hy;
                            break;
                        case 3:  // flux at T
                            if (k > 0)
                                aF[k][j][i]   += flux / hz;
                            if (k < info->mz-1 && k+1 < info->zs + info->zm)
                                aF[k+1][j][i] -= flux / hz;
                            break;
                    }
                }
            }
        }
    }
    return 0;
}

