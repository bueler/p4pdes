static char help[] =
"Structured-grid Poisson problem in 1D, 2D, or 3D.  Option prefix fsh_.\n"
"Solves  - nabla^2 u = f  by putting it in form  F(u) = - nabla^2 u - f.\n"
"Dirichlet boundary conditions on unit square.  Three different problems\n"
"where exact solution is known.  Uses DMDA and SNES.  Multigrid-capable\n"
"because call-backs fully-rediscretize for the supplied grid.  Defaults\n"
"to 2D.\n\n";

/*
# generate .m files with solutions at levels:
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -snes_monitor_solution ascii:u.m:ascii_matlab
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_3_ksp_monitor_solution ascii:errlevel3.m:ascii_matlab
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_2_ksp_monitor_solution ascii:errlevel2.m:ascii_matlab
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_levels_1_ksp_monitor_solution ascii:errlevel1.m:ascii_matlab

because default -mg_coarse_ksp_type is preonly, without changing that we get nothing:
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_coarse_ksp_type cg -snes_monitor -ksp_converged_reason -mg_levels_ksp_monitor -mg_coarse_ksp_monitor|less
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 3 -pc_type mg -ksp_rtol 1.0e-12 -mg_coarse_ksp_type cg -mg_coarse_ksp_monitor_solution ascii:errcoarse.m:ascii_matlab

FD jacobian with coloring is actually faster (and it is clear what is going on):
$ timer ./fish -fsh_dim 1 -fsh_problem manupoly -snes_monitor -da_refine 16
$ timer ./fish -fsh_dim 1 -fsh_problem manupoly -snes_monitor -da_refine 16 -snes_fd_color

compare whether rediscretization happens at each level (former) or Galerkin grid-
transfer operators are used (latter)
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 4 -pc_type mg -snes_monitor
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 4 -pc_type mg -snes_monitor -pc_mg_galerkin

choose linear solver for coarse grid (default is preonly+lu):
$ ./fish -fsh_dim 1 -fsh_problem manupoly -da_refine 4 -pc_type mg -mg_coarse_ksp_type cg -mg_coarse_pc_type jacobi -ksp_view|less

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

double u_exact_1Dmanupoly(double x, double y, double z, void *ctx) {
    return x*x * (1.0 - x*x);
}

double u_exact_2Dmanupoly(double x, double y, double z, void *ctx) {
    return x*x * (1.0 - x*x) * y*y *(y*y - 1.0);
}

double u_exact_3Dmanupoly(double x, double y, double z, void *ctx) {
    return x*x * (1.0 - x*x) * y*y * (y*y - 1.0) * z*z * (z*z - 1.0);
}

double u_exact_1Dmanuexp(double x, double y, double z, void *ctx) {
    return - exp(x);
}

double u_exact_2Dmanuexp(double x, double y, double z, void *ctx) {
    return - x * exp(y);
}

double u_exact_3Dmanuexp(double x, double y, double z, void *ctx) {
    return - x * exp(y + z);
}

double zero(double x, double y, double z, void *ctx) {
    return 0.0;
}

// right-hand-side functions  f(x,y) = - laplacian u

double f_rhs_1Dmanupoly(double x, double y, double z, void *ctx) {
    return 12.0 * x*x - 2.0;
}

double f_rhs_2Dmanupoly(double x, double y, double z, void *ctx) {
    double aa, bb, ddaa, ddbb;
    aa = x*x * (1.0 - x*x);
    bb = y*y * (y*y - 1.0);
    ddaa = 2.0 * (1.0 - 6.0 * x*x);
    ddbb = 2.0 * (6.0 * y*y - 1.0);
    return - (ddaa * bb + aa * ddbb);
}

double f_rhs_3Dmanupoly(double x, double y, double z, void *ctx) {
    double aa, bb, cc, ddaa, ddbb, ddcc;
    aa = x*x * (1.0 - x*x);
    bb = y*y * (y*y - 1.0);
    cc = z*z * (z*z - 1.0);
    ddaa = 2.0 * (1.0 - 6.0 * x*x);
    ddbb = 2.0 * (6.0 * y*y - 1.0);
    ddcc = 2.0 * (6.0 * z*z - 1.0);
    return - (ddaa * bb * cc + aa * ddbb * cc + aa * bb * ddcc);
}

double f_rhs_1Dmanuexp(double x, double y, double z, void *ctx) {
    return exp(x);
}

double f_rhs_2Dmanuexp(double x, double y, double z, void *ctx) {
    return x * exp(y);  // note  f = - (u_xx + u_yy) = - u
}

double f_rhs_3Dmanuexp(double x, double y, double z, void *ctx) {
    return 2.0 * x * exp(y + z);  // note  f = - laplacian u = - 2 u
}


// functions simply to put u_exact()=g_bdry() into a grid; irritatingly-dimension-dependent

PetscErrorCode Form1DUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
  PetscErrorCode ierr;
  int          i;
  double       xmax[1], xmin[1], hx, x, *au;
  ierr = DMDAGetBoundingBox(info->da,xmin,xmax); CHKERRQ(ierr);
  hx = (xmax[0] - xmin[0]) / (info->mx - 1);
  ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
  for (i=info->xs; i<info->xs+info->xm; i++) {
      x = xmin[0] + i * hx;
      au[i] = user->g_bdry(x,0.0,0.0,user);
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
            au[j][i] = user->g_bdry(x,y,0.0,user);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode Form3DUExact(DMDALocalInfo *info, Vec u, PoissonCtx* user) {
    PetscErrorCode ierr;
    int    i, j, k;
    double xyzmin[3], xyzmax[3], hx, hy, hz, x, y, z, ***au;
    ierr = DMDAGetBoundingBox(info->da,xyzmin,xyzmax); CHKERRQ(ierr);
    hx = (xyzmax[0] - xyzmin[0]) / (info->mx - 1);
    hy = (xyzmax[1] - xyzmin[1]) / (info->my - 1);
    hz = (xyzmax[2] - xyzmin[2]) / (info->mz - 1);
    ierr = DMDAVecGetArray(info->da, u, &au);CHKERRQ(ierr);
    for (k=info->zs; k<info->zs+info->zm; k++) {
        z = xyzmin[2] + k * hz;
        for (j=info->ys; j<info->ys+info->ym; j++) {
            y = xyzmin[1] + j * hy;
            for (i=info->xs; i<info->xs+info->xm; i++) {
                x = xyzmin[0] + i * hx;
                au[k][j][i] = user->g_bdry(x,y,z,user);
            }
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
    return 0;
}

//STARTPTRARRAYS
// arrays of pointers to functions:   ..._ptr[DIMS]
static void* residual_ptr[3]
    = {&Form1DFunctionLocal, &Form2DFunctionLocal, &Form3DFunctionLocal};

static void* jacobian_ptr[3]
    = {&Form1DJacobianLocal, &Form2DJacobianLocal, &Form3DJacobianLocal};

static void* getuexact_ptr[3]
    = {&Form1DUExact, &Form2DUExact, &Form3DUExact};
//ENDPTRARRAYS

typedef enum {MANUPOLY, MANUEXP, ZERO} ProblemType;
static const char* ProblemTypes[] = {"manupoly","manuexp","zero",
                                     "ProblemType", "", NULL};

// more arrays of pointers to functions:   ..._ptr[DIMS][PROBLEMS]

static void* g_bdry_ptr[3][3]
    = {{&u_exact_1Dmanupoly, &u_exact_1Dmanuexp, &zero},
       {&u_exact_2Dmanupoly, &u_exact_2Dmanuexp, &zero},
       {&u_exact_3Dmanupoly, &u_exact_3Dmanuexp, &zero}};

static void* f_rhs_ptr[3][3]
    = {{&f_rhs_1Dmanupoly, &f_rhs_1Dmanuexp, &zero},
       {&f_rhs_2Dmanupoly, &f_rhs_2Dmanuexp, &zero},
       {&f_rhs_3Dmanupoly, &f_rhs_3Dmanuexp, &zero}};


int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    Vec            u, uexact;
    PoissonCtx     user;
    DMDALocalInfo  info;
    double         errinf, normconst2h, err2h;
    char           gridstr[99];
    PetscErrorCode (*getuexact)(DMDALocalInfo*,Vec,PoissonCtx*);
    // defaults:
    int            dim = 2;                       // 2D
    ProblemType    problem = MANUEXP;             // manufactured problem using exp()
    PetscBool      init_random = PETSC_FALSE,     // default to zero initial iterate
                   q0interp = PETSC_FALSE;
    double         Lx = 1.0, Ly = 1.0, Lz = 1.0;  // domain [0,1]x[0,1]x[0,1]

    PetscInitialize(&argc,&argv,NULL,help);
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"fsh_", "options for fish.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",
         "dimension of problem (=1,2,3 only)",
         "fish.c",dim,&dim,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Lx",
         "set Lx in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",Lx,&Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Ly",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",Ly,&Ly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Lz",
         "set Ly in domain ([0,Lx] x [0,Ly] x [0,Lz], etc.)",
         "fish.c",Lz,&Lz,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-init_random",
         "initial state is random (default is zero)",
         "fish.c",init_random,&init_random,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem",
         "problem type (determines exact solution and RHS)",
         "fish.c",ProblemTypes,(PetscEnum)problem,(PetscEnum*)&problem,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-q0interp",
         "use DMDAInterpolationType of DMDA_Q0 instead of default DMDA_Q1",
         "fish.c",q0interp,&q0interp,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);
    user.g_bdry = g_bdry_ptr[dim-1][problem];
    user.f_rhs = f_rhs_ptr[dim-1][problem];

//STARTCREATE
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
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,1,"invalid dim for DMDA creation\n");
    }
    if (q0interp) {
        DMDASetInterpolationType(da, DMDA_Q0);
    }

    ierr = DMSetApplicationContext(da,&user); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // call BEFORE SetUniformCoordinates
    ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,Lz); CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)(residual_ptr[dim-1]),&user); CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)(jacobian_ptr[dim-1]),&user); CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);
//ENDCREATE

    ierr = DMCreateGlobalVector(da,&u); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u"); CHKERRQ(ierr);
    if (init_random) {
        PetscRandom   rctx;
        ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx); CHKERRQ(ierr);
        ierr = VecSetRandom(u,rctx); CHKERRQ(ierr);
        ierr = PetscRandomDestroy(&rctx); CHKERRQ(ierr);
    } else {
        ierr = VecSet(u,0.0); CHKERRQ(ierr);
    }

    ierr = SNESSolve(snes,NULL,u); CHKERRQ(ierr);

    ierr = VecDuplicate(u,&uexact); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    getuexact = getuexact_ptr[dim-1];
    ierr = (*getuexact)(&info,uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&err2h); CHKERRQ(ierr);

    switch (dim) {
        case 1:
            normconst2h = PetscSqrtReal((double)(info.mx-1));
            snprintf(gridstr,99,"%d point 1D",info.mx);
            break;
        case 2:
            normconst2h = PetscSqrtReal((double)(info.mx-1)*(info.my-1));
            snprintf(gridstr,99,"%d x %d point 2D",info.mx,info.my);
            break;
        case 3:
            normconst2h = PetscSqrtReal((double)(info.mx-1)*(info.my-1)*(info.mz-1));
            snprintf(gridstr,99,"%d x %d x %d point 3D",info.mx,info.my,info.mz);
            break;
        default:
            SETERRQ(PETSC_COMM_WORLD,4,"invalid dim value in final report\n");
    }
    err2h /= normconst2h; // like continuous L2
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                "problem %s on %s grid:\n"
                "  error |u-uexact|_inf = %.3e, |u-uexact|_h = %.3e\n",
                ProblemTypes[problem],gridstr,errinf,err2h); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);  SNESDestroy(&snes);  DMDestroy(&da);
    return PetscFinalize();
}

