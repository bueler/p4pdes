static const char help[] =
"Solves elasto-plastic torsion problem in 2D using SNESVI.  Option prefix -el_.\n"
"Equation is  - grad^2 u = f(x,y)  where  f(x,y) = 2 C.  Zero Dirichlet\n"
"boundary conditions.  Domain Omega = (0,1)x(0,1).  Constraint is upper bound\n"
"   u(x,y) <= dist((x,y), bdry Omega) = psi(x,y).\n"
"At locations where the constraint is active, the material experiences plastic\n"
"failure.  Where inactive, the equation represents the elasticity.  As with\n"
"related codes ../obstacle.c and dam.c the code reuses the residual and\n"
"Jacobian evaluation code from ch6/.\n"
"Reference: R. Kornhuber (1994) 'Monotone multigrid methods for elliptic\n"
"variational inequalities I', Numerische Mathematik, 69(2), 167-184.\n\n";

#include <petsc.h>
#include "../../ch6/poissonfunctions.h"

// z = psi(x,y) = dist((x,y), bdry Omega)  is the upper obstacle
PetscReal psi(PetscReal x, PetscReal y) {
    if (y <= x)
        if (y <= 1.0 - x)
            return y;
        else
            return 1.0 - x;
    else
        if (y <= 1.0 - x)
            return x;
        else
            return 1.0 - y;
}

PetscReal zero(PetscReal x, PetscReal y, PetscReal z, void *ctx) {
    return 0.0;
}

typedef struct {
    PetscReal  C;   // physical parameter
} ElastoCtx;

PetscReal f_fcn(PetscReal x, PetscReal y, PetscReal z, void *user) {
    PoissonCtx *poi = (PoissonCtx*) user;
    ElastoCtx  *elasto = (ElastoCtx*) poi->addctx;
    return 2.0 * elasto->C;
}

extern PetscErrorCode FormBounds(SNES, Vec, Vec);

int main(int argc,char **argv) {
  DM                   da, da_after;
  SNES                 snes;
  Vec                  u_initial, u;
  PoissonCtx           user;
  ElastoCtx            elasto;
  SNESConvergedReason  reason;
  PetscInt             snesits;
  PetscReal            lflops,flops;
  DMDALocalInfo        info;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  elasto.C = 2.5;
  PetscOptionsBegin(PETSC_COMM_WORLD,"el_",
                           "elasto-plastic torsion solver options","");
  PetscCall(PetscOptionsReal("-C","f(x,y)=2C is source term",
                          "elasto.c",elasto.C,&elasto.C,NULL));
  PetscOptionsEnd();

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      3,3,                       // override with -da_grid_x,_y
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0));

  user.cx = 1.0;
  user.cy = 1.0;
  user.cz = 1.0;
  user.g_bdry = &zero;
  user.f_rhs = &f_fcn;
  user.addctx = &elasto;
  PetscCall(DMSetApplicationContext(da,&user));

  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));
  PetscCall(SNESSetDM(snes,da));
  PetscCall(SNESSetApplicationContext(snes,&user));

  PetscCall(SNESSetType(snes,SNESVINEWTONRSLS));
  PetscCall(SNESVISetComputeVariableBounds(snes,&FormBounds));

  // reuse residual and jacobian from ch6/
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)Poisson2DFunctionLocal,&user));
  PetscCall(DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Poisson2DJacobianLocal,&user));
  PetscCall(SNESSetFromOptions(snes));

  // initial iterate is zero
  PetscCall(DMCreateGlobalVector(da,&u_initial));
  PetscCall(VecSet(u_initial,0.0));

  /* solve; then get solution and DM after solution*/
  PetscCall(SNESSolve(snes,NULL,u_initial));
  PetscCall(VecDestroy(&u_initial));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESGetDM(snes,&da_after));
  PetscCall(SNESGetSolution(snes,&u)); /* do not destroy u */

  /* performance measures */
  PetscCall(SNESGetConvergedReason(snes,&reason));
  if (reason <= 0) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
          "WARNING: SNES not converged ... use -snes_converged_reason to check\n"));
  }
  PetscCall(SNESGetIterationNumber(snes,&snesits));
  PetscCall(PetscGetFlops(&lflops));
  PetscCall(MPI_Allreduce(&lflops,&flops,1,MPIU_REAL,MPIU_SUM,PETSC_COMM_WORLD));
  PetscCall(DMDAGetLocalInfo(da_after,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "done on %d x %d grid using C=%.3f: total flops = %.3e; SNES iterations %d\n",
      info.mx,info.my,elasto.C,flops,snesits));

  SNESDestroy(&snes);
  PetscCall(PetscFinalize());
  return 0;
}


// for call-back: tell SNESVI we want  -infty < u <= psi
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i, j;
  PetscReal      **aXu, dx, dy, x, y;
  PetscCall(VecSet(Xu,PETSC_NINFINITY));
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMDAGetLocalInfo(da,&info));
  dx = 1.0 / (PetscReal)(info.mx-1);
  dy = 1.0 / (PetscReal)(info.my-1);
  PetscCall(DMDAVecGetArray(da, Xu, &aXu));
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * dx;
      aXu[j][i] = psi(x,y);
    }
  }
  PetscCall(DMDAVecRestoreArray(da, Xu, &aXu));
  return 0;
}
