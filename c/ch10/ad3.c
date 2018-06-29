static char help[] =
"Solves a 3D linear advection-diffusion problem using FD discretization,\n"
"structured-grid (DMDA), and -snes_fd_color.  Option prefix -ad3_.  Equation:\n"
"    - eps Laplacian u + div (a(x,y,z) u) = g(x,y,z),\n"
"where the wind a(x,y,z) and source g(x,y,z) are given smooth functions.\n"
"The domain is [-1,1]^3 with Dirichlet and periodic boundary conditions:\n"
"    u(1,y,z) = b(y,z)\n"
"    u(-1,y,z) = u(x,y,-1) = u(x,y,1) = 0\n"
"    u periodic in y\n"
"where b(y,z) is a given smooth function.  Problems include: LAYER = (exact\n"
"solution based on a boundary layer of width eps), NOWIND = (exact\n"
"diffusion-only solution), and GLAZE = (double-glazing problem).  Advection\n"
"can be discretized by first-order upwinding, centered, or van Leer\n"
"limiter schemes.\n\n";

/* TODO:
2. create LAYERX and LAYERZ problems
3. visualize glaze problem to check for correctness
4. implement Jacobian for -ad3_limiter none; test with -snes_mf_operator -ad3_limiter vanleer
*/

/* shows scaling is on the dot so that GMG has constant its, and converges, for NOWIND:
for LEV in 1 2 3 4; do ./ad3 -ad3_problem nowind -ad3_limiter none -da_refine $LEV -ksp_converged_reason -ksp_type cg -pc_type mg; done
compare in ch6/:
for LEV in 1 2 3 4; do ./fish -fsh_dim 3 -da_grid_x 6 -da_grid_y 7 -da_grid_z 6 -da_refine $LEV -ksp_converged_reason -snes_type ksponly -ksp_type cg -pc_type mg; done
*/

/* acting like it is correct for LAYER with easy eps=1.0 and using GMG with ILU smoothing:
for LIM in none centered vanleer; do
    for LEV in 1 2 3 4; do
        timer ./ad3 -ad3_limiter $LIM -ksp_converged_reason -da_refine $LEV -ksp_rtol 1.0e-9 -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu
    done
done
(going to LEV=5 generates seg fault from attempt to get too much memory?)
*/

/* fastest GMG method for significant advection-dominated:
timer ./ad3 -ad3_limiter none -ksp_converged_reason -da_refine LEV -ksp_rtol 1.0e-9 -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -ksp_monitor -ad3_eps 0.1 -ksp_type bcgs
can go to LEV=6 on ed-galago?
*/

/* there is no need to refine in y-direction; this refinement path allows
fully-resolving the boundary layer in LAYER:
for LEV in 1 2 3 4 5 6 7; do
    timer ./ad3 -ad3_eps 0.01 -ad3_limiter none -snes_type ksponly -ksp_converged_reason -pc_type mg -mg_levels_ksp_type richardson -mg_levels_pc_type ilu -da_grid_y 21 -da_refine_y 1 -da_refine $LEV
done
  * replace with "-da_grid_y 20" for centered, vanleer
  * could be improved by having fully y-independent exact solution
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

typedef enum {LAYER, NOWIND, GLAZE} ProblemType;
static const char *ProblemTypes[] = {"layer", "nowind", "glaze",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType problem;
    double      eps,          // amount of diffusion; eps > 0
                glaze_drift;
    double      (*limiter_fcn)(double),
                (*g_fcn)(double, double, double, void*),
                (*b_fcn)(double, double, void*);
} AdCtx;

/* problem LAYER:
A partly-manufactured 3D exact solution of a boundary layer problem, with
exponential layer near x=1:
    u(x,y,z) = U(x) sin(E (y+1)) sin(F (z+1))
where
    U(x) = (exp((x+1) / eps) - 1) / (exp(2 / eps) - 1)
         = (exp((x-1) / eps) - C) / (1 - C)
where  C = exp(-2 / eps).  (Note C may gracefully underflow (not overflow)
if eps is really small.)  Thus U(x) satisfies
    -eps U'' + U' = 0,
U(-1)=0, U(1)=1, and it has a boundary layer of width O(eps) near x=1.
Constants E = 2 pi and F = pi / 2 are set so that u is periodic
and smooth in y and satisfies Dirichlet boundary conditions in z (i.e.
u(x,y,+-1) = 0.)  The problem solved has
    a = <1,0,0>
    g(x,y,z,u) = lambda u
where lambda = eps (E^2 + F^2), and
    b(y,z) = u(1,y,z)
*/

static const double EE = 2.0 * PETSC_PI,
                    FF = PETSC_PI / 2.0;

static double layer_u(double x, double y, double z, AdCtx *user) {
    const double C = exp(-2.0 / user->eps); // may underflow to 0; that's o.k.
    return ((exp((x-1) / user->eps) - C) / (1.0 - C))
           * sin(EE*(y+1.0)) * sin(FF*(z+1.0));
}

static double layer_g(double x, double y, double z, void *user) {
    AdCtx* usr = (AdCtx*)user;
    const double lam = usr->eps * (EE*EE + FF*FF);
    return lam * layer_u(x,y,z,usr);
}

static double layer_b(double y, double z, void *user) {
    return layer_u(1.0,y,z,(AdCtx*)user);
}

/* problem NOWIND:
A manufactured 3D exact solution of a boundary layer problem for testing the
diffusion only problem:
    u(x,y,z) = sin(A (x+1)) sin(E (y+1)) sin(F (z+1))
Constants E = 2 pi and F = pi / 2 as in LAYER, but A = pi / 4.  Problem has
    a = <0,0,0>
    g(x,y,z,u) = mu u
where mu = eps (A^2 + E^2 + F^2), and
    b(y,z) = u(+1,y,z)
*/

static const double AA = PETSC_PI / 4.0;

static double nowind_u(double x, double y, double z, AdCtx *user) {
    return sin(AA*(x+1.0)) * sin(EE*(y+1.0)) * sin(FF*(z+1.0));
}

static double nowind_g(double x, double y, double z, void *user) {
    AdCtx* usr = (AdCtx*)user;
    const double mu = usr->eps * (AA*AA + EE*EE + FF*FF);
    return mu * nowind_u(x,y,z,usr);
}

static double nowind_b(double y, double z, void *user) {
    return nowind_u(1.0,y,z,(AdCtx*)user);
}

/* problem GLAZE:
See pages 240-241 of Elman et al (2014) for this problem, a recirculating flow
which is counterclockwise in the x-z plane.  Note g(x,y,z) = 0 and b(y,z) = 1.
See wind_a() for velocity, and not an additional drift can be added in the
y-direction.
*/

static double glaze_g(double x, double y, double z, void *user) {
    return 0.0;
}

static double glaze_b(double y, double z, void *user) {
    return 1.0;
}

static void* gptr[] = {&layer_g, &nowind_g, &glaze_g};
static void* bptr[] = {&layer_b, &nowind_b, &glaze_b};

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

