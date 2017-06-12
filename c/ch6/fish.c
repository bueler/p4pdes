static char help[] =
"Structured-grid Poisson problem in 1D, 2D, or 3D.  Option prefix fsh_.\n"
"Solves  - nabla^2 u = f  by putting it in form  F(u) = - nabla^2 u - f.\n"
"Dirichlet boundary conditions on unit square.  Three different problems\n"
"where exact solution is known.  Uses DMDA and SNES.  Multigrid-capable\n"
"because call-backs fully-rediscretize for the supplied grid.  Defaults\n"
"to 2D.\n\n";

/*
note -fsh_dim 2 -fsh_problem manuexp is problem re-used many times in Smith et al 1996

add options to any run:
    -{ksp,snes}_monitor -{ksp,snes}_converged_reason

since these are linear problems, consider adding:
    -snes_type ksponly -ksp_rtol 1.0e-12

see study/mgstudy.sh for multigrid parameter study

this makes sense and shows V-cycles:
$ ./fish -fsh_dim 2 -da_refine 3 -pc_type mg -snes_type ksponly -ksp_converged_reason -mg_levels_ksp_monitor

in parallel with -snes_fd_color (exploits full rediscretization)
$ mpiexec -n 2 ./fish -fsh_dim 2 -da_refine 4 -pc_type mg -snes_fd_color

compare with rediscretization at every level or use Galerkin coarse grid operator
$ ./fish -fsh_dim 2 -da_refine 4 -pc_type mg -snes_monitor
$ ./fish -fsh_dim 2 -da_refine 4 -pc_type mg -snes_monitor -pc_mg_galerkin

choose linear solver for coarse grid (default is preonly+lu):
$ ./fish -fsh_dim 2 -da_refine 4 -pc_type mg -mg_coarse_ksp_type cg -mg_coarse_pc_type jacobi -ksp_view

to make truly random init, with time as seed, add
    #include <time.h>
    ...
        ierr = PetscRandomSetSeed(rctx,time(NULL)); CHKERRQ(ierr);
        ierr = PetscRandomSeed(rctx); CHKERRQ(ierr);

to generate classical jacobi/gauss-seidel results, put f in a Vec and
add viewer for RHS:
   PetscViewer viewer;
   PetscViewerASCIIOpen(COMM,"rhs.m",&viewer);
   PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
   VecView(f,viewer);
then do:
$ ./fish -fsh_dim 2 -da_refine 1 -snes_monitor -ksp_monitor -snes_max_it 1 -ksp_type richardson -pc_type jacobi|sor
with e.g. -ksp_monitor_solution :foo.m:ascii_matlab
*/

#include <petsc.h>
#include "poissonfunctions.h"


// exact solutions  u(x,y),  for boundary condition and error calculation

double u_exact_1Dmanupoly(double x, double y, double z) {
    return x*x * (1.0 - x*x);
}

double u_exact_2Dmanupoly(double x, double y, double z) {
    return (x - x*x) * (y*y - y);
}

double u_exact_2Dmanuexp(double x, double y, double z) {
    return - x * exp(y);
}

double u_exact_zero(double x, double y, double z) {
    return 0.0;
}

// right-hand-side functions  f(x,y) = - laplacian u

double f_rhs_1Dmanupoly(double x, double y, double z) {
    return 12.0 * x*x - 2.0;
}

double f_rhs_2Dmanupoly(double x, double y, double z) {
    double uxx, uyy;
    uxx  = - 2.0 * (y*y - y);
    uyy  = (x - x*x) * 2.0;
    return - uxx - uyy;
}

double f_rhs_2Dmanuexp(double x, double y, double z) {
    return x * exp(y);  // indeed   - (u_xx + u_yy) = -u  !
}

double f_rhs_zero(double x, double y, double z) {
    return 0.0;
}


PetscErrorCode Form1DUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
  PetscErrorCode ierr;
  int          i;
  double       xmax[1], xmin[1], hx, x, *au;
  ierr = DMDAGetBoundingBox(info->da,xmin,xmax); CHKERRQ(ierr);
  hx = (xmax[0] - xmin[0]) / (info->mx - 1);
  ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
  for (i=info->xs; i<info->xs+info->xm; i++) {
      x = xmin[0] + i * hx;
      au[i] = user->u_exact(x,0.0,0.0);
  }
  ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode Form2DUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
    PetscErrorCode ierr;
    int     i, j;
    double  xymin[2], xymax[2], hx, hy, x, y, **au;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
    for (j=info->ys; j<info->ys+info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i=info->xs; i<info->xs+info->xm; i++) {
            x = xymin[0] + i * hx;
            au[j][i] = user->u_exact(x,y,0.0);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
    return 0;
}

// arrays of pointers to functions:   ..._ptr[DIMS]
static void* residual_ptr[3]
    = {&Form1DFunctionLocal, &Form2DFunctionLocal, &Form3DFunctionLocal};

static void* jacobian_ptr[3]
    = {&Form1DJacobianLocal, &Form2DJacobianLocal, &Form3DJacobianLocal};

static void* getuexact_ptr[3]
    = {&Form1DUExact, &Form2DUExact, NULL};

typedef enum {MANUPOLY, MANUEXP, ZERO} ProblemType;
static const char* ProblemTypes[] = {"manupoly","manuexp","zero",
                                     "ProblemType", "", NULL};

// more arrays of pointers to functions:   ..._ptr[DIMS][CASES]

static void* u_exact_ptr[3][3]
    = {{&u_exact_1Dmanupoly, NULL,               &u_exact_zero},
       {&u_exact_2Dmanupoly, &u_exact_2Dmanuexp, &u_exact_zero},
       {NULL,                NULL,               &u_exact_zero}};

static void* f_rhs_ptr[3][3]
    = {{&f_rhs_1Dmanupoly, NULL,             &f_rhs_zero},
       {&f_rhs_2Dmanupoly, &f_rhs_2Dmanuexp, &f_rhs_zero},
       {NULL,              NULL,             &f_rhs_zero}};

int main(int argc,char **argv) {
    PetscErrorCode    ierr;
    DM             da;
    SNES           snes;
    KSP            ksp;
    Vec            u, uexact;
    PoissonCtx     user;
    ProblemType    problem = MANUEXP;
    int            dim = 2;
    double         Lx = 1.0, Ly = 1.0, Lz = 1.0;
    PetscBool      init_random = PETSC_FALSE;
    DMDALocalInfo  info;
    double         errinf, normconst2h, err2h;
    double         (*getuexact)(DMDALocalInfo*,Vec,PoissonCtx*);

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"fsh_", "options for fish.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",
         "dimension of problem (=1,2,3 only)","fish.c",dim,&dim,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Lx",
         "set Lx in domain ([0,Lx] x [0,Ly] x [0,Lz], or for lower dim)","fish.c",Lx,&Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Ly",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], or for lower dim)","fish.c",Ly,&Ly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Lz",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], or for lower dim)","fish.c",Lz,&Lz,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-init_random",
         "initial state is random (default is zero)",
         "fish.c",init_random,&init_random,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem",
         "problem type (determines exact solution and RHS)",
         "fish.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    switch (dim) {
        case 1:
            ierr = DMDACreate1d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE,3,1,1, NULL, &da); CHKERRQ(ierr);
            break;
        case 2:
            ierr = DMDACreate2d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da); CHKERRQ(ierr);
            break;
        case 3:
            ierr = DMDACreate3d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,
                3,3,3,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
                1,1,NULL,NULL,NULL,&da); CHKERRQ(ierr);
        default:
            SETERRQ(PETSC_COMM_WORLD,1,"invalid dim value in creating DMDA\n");
    }

    user.u_exact = u_exact_ptr[dim-1][problem];
    user.f_rhs = f_rhs_ptr[dim-1][problem];
    getuexact = getuexact_ptr[dim-1];
    if (user.u_exact == NULL) {
        SETERRQ(PETSC_COMM_WORLD,2,"error setting up u_exact() function\n");
    }
    if (user.f_rhs == NULL) {
        SETERRQ(PETSC_COMM_WORLD,3,"error setting up f_rhs() function\n");
    }
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)(residual_ptr[dim-1]),&user); CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)(jacobian_ptr[dim-1]),&user); CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp); CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPCG); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

    if (init_random) {
        PetscRandom   rctx;
        ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx); CHKERRQ(ierr);
        ierr = VecSetRandom(u,rctx); CHKERRQ(ierr);
        ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);
    } else {
        ierr = VecSet(u,0.0); CHKERRQ(ierr);
    }

    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = getuexact(&info,uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&err2h); CHKERRQ(ierr);

    switch (dim) {
        case 1:
            normconst2h = PetscSqrtReal((double)(info.mx-1));
            err2h /= normconst2h; // like continuous L2
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "on %d point grid:  error |u-uexact|_inf = %g, |...|_h = %.2e\n",
                info.mx,errinf,err2h); CHKERRQ(ierr);
            break;
        case 2:
            normconst2h = PetscSqrtReal((double)(info.mx-1)*(info.my-1));
            err2h /= normconst2h; // like continuous L2
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "on %d x %d grid:  error |u-uexact|_inf = %g, |...|_h = %.2e\n",
                info.mx,info.my,errinf,err2h); CHKERRQ(ierr);
            break;
        case 3:
            normconst2h = PetscSqrtReal((double)(info.mx-1)*(info.my-1)*(info.mz-1));
            err2h /= normconst2h; // like continuous L2
            ierr = PetscPrintf(PETSC_COMM_WORLD,
                "on %d x %d x %d grid:  error |u-uexact|_inf = %g, |...|_h = %.2e\n",
                info.mx,info.my,info.mz,errinf,err2h); CHKERRQ(ierr);
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,4,"invalid dim value in final report\n");
    }

    VecDestroy(&u);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

