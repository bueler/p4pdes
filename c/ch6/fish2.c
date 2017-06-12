static char help[] =
"Structured-grid Poisson problem in 2D using DMDA+SNES.  Option prefix fsh_.\n"
"Solves  - nabla^2 u = f  by putting it in form  F(u) = - nabla^2 u - f.\n"
"Dirichlet boundary conditions on unit square.  Three different problems\n"
"where exact solution is known.  Multigrid-capable because call-backs\n"
"fully-rediscretize for the supplied grid.\n\n";

/*
note -fsh_problem manuexp is problem re-used many times in Smith et al 1996

add options to any run:
    -{ksp,snes}_monitor -{ksp,snes}_converged_reason

since these are linear problems, consider adding:
    -snes_type ksponly -ksp_rtol 1.0e-12

see study/mgstudy.sh for multigrid parameter study

this makes sense and shows V-cycles:
$ ./fish2 -da_refine 3 -pc_type mg -snes_type ksponly -ksp_converged_reason -mg_levels_ksp_monitor

in parallel with -snes_fd_color (exploits full rediscretization)
$ mpiexec -n 2 ./fish2 -da_refine 4 -pc_type mg -snes_fd_color

compare with rediscretization at every level or use Galerkin coarse grid operator
$ ./fish2 -da_refine 4 -pc_type mg -snes_monitor
$ ./fish2 -da_refine 4 -pc_type mg -snes_monitor -pc_mg_galerkin

choose linear solver for coarse grid (default is preonly+lu):
$ ./fish2 -da_refine 4 -pc_type mg -mg_coarse_ksp_type cg -mg_coarse_pc_type jacobi -ksp_view

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
$ ./fish2 -da_refine 1 -snes_monitor -ksp_monitor -snes_max_it 1 -ksp_type richardson -pc_type jacobi|sor
with e.g. -ksp_monitor_solution :foo.m:ascii_matlab
*/

#include <petsc.h>
#include "../jacobians.h"

typedef enum {MANUPOLY, MANUEXP, ZERO} ProblemType;
static const char *ProblemTypes[] = {"manupoly","manuexp","zero",
                                     "ProblemType", "", NULL};

typedef struct {
    ProblemType problem;
} FishCtx;

// the exact solution u(x,y), for boundary condition and error calculation
double u_exact(double x, double y, ProblemType problem) {
    switch (problem) {
        case MANUPOLY:
            return (x - x*x) * (y*y - y);
        case MANUEXP:
            return - x * exp(y);
        default:
            return 0.0;
    }
}

// the right-hand-side f(x,y) = - laplacian u
double f_rhs(double x, double y, ProblemType problem) {
    switch (problem) {
        case MANUPOLY: {
            double uxx, uyy;
            uxx  = - 2.0 * (y*y - y);
            uyy  = (x - x*x) * 2.0;
            return - uxx - uyy;
        }
        case MANUEXP:
            return x * exp(y); // indeed   - (u_xx + u_yy) = -u  !
        default:
            return 0.0;
    }
}

PetscErrorCode formExact(DMDALocalInfo *info, Vec u, FishCtx* user) {
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
            au[j][i] = u_exact(x,y,user->problem);
        }
    }
    ierr = DMDAVecRestoreArray(info->da, u, &au);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, double **au,
                                 double **FF, FishCtx *user) {
    PetscErrorCode ierr;
    int     i, j;
    double  hx, hy, xymin[2], xymax[2], x, y, ue, uw, un, us;
    ierr = DMDAGetBoundingBox(info->da,xymin,xymax); CHKERRQ(ierr);
    hx = (xymax[0] - xymin[0]) / (info->mx - 1);
    hy = (xymax[1] - xymin[1]) / (info->my - 1);
    for (j = info->ys; j < info->ys + info->ym; j++) {
        y = xymin[1] + j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++) {
            x = xymin[0] + i * hx;
            if (i==0 || i==info->mx-1 || j==0 || j==info->my-1) {
                FF[j][i] = au[j][i] - u_exact(x,y,user->problem);
            } else {
                ue = (i+1 == info->mx-1) ? u_exact(x+hx,y,user->problem)
                                         : au[j][i+1];
                uw = (i-1 == 0)          ? u_exact(x-hx,y,user->problem)
                                         : au[j][i-1];
                un = (j+1 == info->my-1) ? u_exact(x,y+hy,user->problem)
                                         : au[j+1][i];
                us = (j-1 == 0)          ? u_exact(x,y-hy,user->problem)
                                         : au[j-1][i];
                FF[j][i] = 2.0 * (hy/hx + hx/hy) * au[j][i]
                           - hy/hx * (uw + ue)
                           - hx/hy * (us + un)
                           - hx * hy * f_rhs(x,y,user->problem);
            }
        }
    }
    return 0;
}

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    DM             da;
    SNES           snes;
    KSP            ksp;
    Vec            u, uexact;
    FishCtx        user;
    double         Lx = 1.0, Ly = 1.0;
    PetscBool      init_random = PETSC_FALSE;
    DMDALocalInfo  info;
    double         errinf,err2h;

    PetscInitialize(&argc,&argv,NULL,help);
    user.problem = MANUEXP;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"fsh_", "options for fish2.c", ""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Lx",
         "set Lx in domain [0,Lx] x [0,Ly]","fish2.c",Lx,&Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Ly",
         "set Ly in domain [0,Lx] x [0,Ly]","fish2.c",Ly,&Ly,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-init_random",
         "initial state is random (default is zero)",
         "fish2.c",init_random,&init_random,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-problem",
         "problem type (determines exact solution and RHS)",
         "fish2.c",ProblemTypes,
         (PetscEnum)user.problem,(PetscEnum*)&user.problem,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
               DMDA_STENCIL_STAR,
               3,3,PETSC_DECIDE,PETSC_DECIDE,
               1,1,NULL,NULL,&da); CHKERRQ(ierr);
    ierr = DMSetFromOptions(da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);  // this must be called BEFORE SetUniformCoordinates
    ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
    ierr = DMDASetUniformCoordinates(da,0.0,Lx,0.0,Ly,0.0,1.0);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)u,"u");CHKERRQ(ierr);

    ierr = SNESCreate(PETSC_COMM_WORLD,&snes); CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da); CHKERRQ(ierr);
    ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)FormFunctionLocal,&user); CHKERRQ(ierr);
    ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Form2DJacobianLocal,&user); CHKERRQ(ierr);
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

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
    ierr = formExact(&info,uexact,&user); CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uexact
    ierr = VecNorm(u,NORM_INFINITY,&errinf); CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_2,&err2h); CHKERRQ(ierr);
    err2h /= PetscSqrtReal((double)(info.mx-1)*(info.my-1)); // like continuous L2
    ierr = PetscPrintf(PETSC_COMM_WORLD,
           "on %d x %d grid:  error |u-uexact|_inf = %g, |...|_h = %.2e\n",
           info.mx,info.my,errinf,err2h); CHKERRQ(ierr);

    VecDestroy(&u);  VecDestroy(&uexact);
    SNESDestroy(&snes);  DMDestroy(&da);
    PetscFinalize();
    return 0;
}

