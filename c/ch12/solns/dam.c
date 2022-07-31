static const char help[] =
"Solves a 2D dam-saturation problem.  No options.\n"
"The exact soluution is not known, but\n"
"a coarse-grid discrete solution can be checked against that source.\n"
"Note Poisson2DFunctionLocal() sets-up this unconstrained problem:\n"
"    - u_xx - u_yy = - 1\n"
"while we want this complementarity problem:\n"
"    F(u) = - u_xx - u_yy + 1 >= 0\n"
"    u >= 0\n"
"    u F(u) = 0.\n"
"As with obstacle.c, this is solved (default) by -snes_type vinewtonrsls.\n"
"Reference:  pages 667-668 of Brandt & Cryer (1983).\n\n";

/*
note PFAS is not implemented in PETSc, but the following runs for X = 1,2,3,4,5,6
quickly solve the same problems as in Brandt & Cryer Table 4.2:
   s ./dam -snes_monitor -pc_type mg -snes_grid_sequence X

on ed-galago I can go up to X = 11 giving 4097 x 6145 grid and serial runtime of 297 seconds
(the next step runs out of memory)

on parallel I am getting error messages with vinewtonrsls + mg:
    mpiexec -n 4 ./dam -snes_converged_reason -snes_grid_sequence 5 -snes_type vinewtonrsls -pc_type mg
but this works with either vinewtonssls or another PC (gamg or bjacobi+ilu or asm+lu or etc.)

run as:
    ./dam -da_refine 1 -snes_view_solution :foo.m:ascii_matlab -snes_converged_reason -snes_rtol 1.0e-12
Nonlinear solve converged due to CONVERGED_FNORM_RELATIVE iterations 4
done on   5 x   7 grid

then in Matlab/Octave you can compare to Brandt & Cryer Table 4.1:
>> foo
>> format long g
>> u = flipud(reshape(Vec_0x84000000_0,5,7)')    % use loaded name here
u =
           0                     0                     0                     0             0
           8      2.53716015237691                     0                     0             0
          32       18.148640609503      6.78414305345868                     0             0
          72      47.2732592321936      24.9879316043211      7.91201621181146             0
         128      89.9564647149903       53.982307919772      22.6601332428759             0
         200      146.570291707977      94.3247021169195      44.7462088399388             0
         288                   218                   148                    78             8

regarding computing seepage face height, runs
    ./dam -snes_converged_reason -pc_type mg -snes_rtol 1.0e-8 -snes_grid_sequence X
gives:
    X    grid       height
    6    129x193    8.8750000
    7    257x385    8.7500000
    8    513x769    8.7500000
    9    1025x1537  8.7343750
    10   2049x3073  8.7187500
    11   4097x6145  8.7109375
but note these numbers depend (at the second digit, even) on the value of
"wetthreshold" in GetSeepageFaceHeight()
*/

#include <petsc.h>
#include "../../ch6/poissonfunctions.h"

typedef struct {
    PetscReal  a, y1, y2;
} DamCtx;

PetscReal g_fcn(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    PoissonCtx       *user = (PoissonCtx*)ctx;
    DamCtx           *dctx = (DamCtx*)(user->addctx);
    const PetscReal  a = dctx->a,
                     y1 = dctx->y1,
                     y2 = dctx->y2,
                     tol = a * 1.0e-8;
    // see (4.2) in Brandt & Cryer for following:
    if (x < tol) {
       return (y1 - y) * (y1 - y) / 2.0;                      // AB
    } else if (x > a - tol) {
       if (y < y2) {
           return (y2 - y) * (y2 - y) / 2.0;                  // CD
       } else {
           return 0.0;                                        // DF
       }
    } else if (y < tol) {
       return (y1 * y1 * (a - x) + y2 * y2 * x) / (2.0 * a);  // BC
    } else if (y > y1 - tol) {
       return 0.0;                                            // FA
    } else {
       return NAN;
    }
}

PetscReal f_fcn(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return -1.0;
}

extern PetscErrorCode FormBounds(SNES, Vec, Vec);
extern PetscErrorCode GetSeepageFaceHeight(DMDALocalInfo*, Vec, PetscReal*, DamCtx*);

int main(int argc,char **argv) {
  DM             da, da_after;
  SNES           snes;
  Vec            u;
  PoissonCtx     user;
  DamCtx         dctx;
  DMDALocalInfo  info;
  PetscReal      height;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  dctx.a  = 16.0;  // a, y1, y2 from Brandt & Cryer
  dctx.y1 = 24.0;
  dctx.y2 = 4.0;
  user.cx = 1.0;
  user.cy = 1.0;
  user.cz = 1.0;
  user.g_bdry = &g_fcn;
  user.f_rhs = &f_fcn;
  user.addctx = &dctx;

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      3,4,                       // override with -da_refine or -da_grid_x,_y
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0.0,dctx.a,0.0,dctx.y1,-1.0,-1.0));
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(SNESSetApplicationContext(snes,&user));

  PetscCall(SNESSetType(snes,SNESVINEWTONRSLS));
  PetscCall(SNESVISetComputeVariableBounds(snes,&FormBounds));
  
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)Poisson2DFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Poisson2DJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateGlobalVector(da,&u));
  // initial iterate has u=g on boundary and u=0 in interior
  PetscCall(InitialState(da, ZEROS, PETSC_TRUE, u, &user));

  /* solve */
  PetscCall(SNESSolve(snes,NULL,u));
  PetscCall(VecDestroy(&u));
  PetscCall(DMDestroy(&da));

  // report seepage face
  PetscCall(SNESGetSolution(snes,&u)); /* do not destroy u */
  PetscCall(SNESGetDM(snes,&da_after));
  PetscCall(DMDAGetLocalInfo(da_after,&info));
  PetscCall(GetSeepageFaceHeight(&info,u,&height,&dctx));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "done on %3d x %3d grid; computed seepage face height = %.7f\n",
      info.mx,info.my,height));

  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

// for call-back: tell SNESVI we want  0 <= u < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
    PetscCall(VecSet(Xl,0.0));
    PetscCall(VecSet(Xu,PETSC_INFINITY));
    return 0;
}

PetscErrorCode GetSeepageFaceHeight(DMDALocalInfo *info, Vec u, PetscReal *height, DamCtx *dctx) {
    MPI_Comm         comm;
    const PetscReal  dy = dctx->y1 / (PetscReal)(info->my-1),
                     wetthreshhold = 1.0e-6;  // what does "u>0" mean?
    PetscInt         j;
    PetscReal         **au, locwetmax = - PETSC_INFINITY;
    PetscCall(DMDAVecGetArrayRead(info->da,u,&au));
    if (info->xs+info->xm == info->mx) { // do we even own (part of) the x=a side of the rectangle?
        for (j=info->ys; j<info->ys+info->ym; j++) {
           if (au[j][info->mx-2] > wetthreshhold) {  // is the first inter point wet?
               locwetmax = PetscMax(j*dy,locwetmax);
           }
        }
    }
    PetscCall(DMDAVecRestoreArrayRead(info->da,u,&au));
    PetscCall(PetscObjectGetComm((PetscObject)(info->da),&comm));
    PetscCall(MPI_Allreduce(&locwetmax,height,1,MPIU_REAL,MPIU_MAX,comm));
    *height -= dctx->y2;   // height is segment ED in figure
    return 0;
}
