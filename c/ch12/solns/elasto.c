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
double psi(double x, double y) {
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

double zero(double x, double y, double z, void *ctx) {
    return 0.0;
}

typedef struct {
    double  C;   // physical parameter
} ElastoCtx;

double f_fcn(double x, double y, double z, void *user) {
    PoissonCtx *poi = (PoissonCtx*) user;
    ElastoCtx  *elasto = (ElastoCtx*) poi->addctx;
    return 2.0 * elasto->C;
}

extern PetscErrorCode FormBounds(SNES, Vec, Vec);

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  DM             da, da_after;
  SNES           snes;
  Vec            u_initial, u;
  PoissonCtx     user;
  ElastoCtx      elasto;
  SNESConvergedReason reason;
  int            snesits;
  double         lflops,flops;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,NULL,help);

  elasto.C = 2.5;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"el_",
                           "elasto-plastic torsion solver options",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-C","f(x,y)=2C is source term",
                          "elasto.c",elasto.C,&elasto.C,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      3,3,                       // override with -da_grid_x,_y
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0);CHKERRQ(ierr);

  user.cx = 1.0;
  user.cy = 1.0;
  user.cz = 1.0;
  user.g_bdry = &zero;
  user.f_rhs = &f_fcn;
  user.addctx = &elasto;
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);

  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);

  // reuse residual and jacobian from ch6/
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
             (DMDASNESFunction)Poisson2DFunctionLocal,&user); CHKERRQ(ierr);
  ierr = DMDASNESSetJacobianLocal(da,
             (DMDASNESJacobian)Poisson2DJacobianLocal,&user); CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // initial iterate is zero
  ierr = DMCreateGlobalVector(da,&u_initial);CHKERRQ(ierr);
  ierr = VecSet(u_initial,0.0); CHKERRQ(ierr);

  /* solve; then get solution and DM after solution*/
  ierr = SNESSolve(snes,NULL,u_initial);CHKERRQ(ierr);
  ierr = VecDestroy(&u_initial); CHKERRQ(ierr);
  ierr = DMDestroy(&da); CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&da_after); CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&u); CHKERRQ(ierr); /* do not destroy u */

  /* performance measures */
  ierr = SNESGetConvergedReason(snes,&reason); CHKERRQ(ierr);
  if (reason <= 0) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,
          "WARNING: SNES not converged ... use -snes_converged_reason to check\n"); CHKERRQ(ierr);
  }
  ierr = SNESGetIterationNumber(snes,&snesits); CHKERRQ(ierr);
  ierr = PetscGetFlops(&lflops); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&lflops,&flops,1,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da_after,&info); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "done on %d x %d grid using C=%.3f: total flops = %.3e; SNES iterations %d\n",
      info.mx,info.my,elasto.C,flops,snesits); CHKERRQ(ierr);

  SNESDestroy(&snes);
  return PetscFinalize();
}


// for call-back: tell SNESVI we want  -infty < u <= psi
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu) {
  PetscErrorCode ierr;
  DM            da;
  DMDALocalInfo info;
  int           i, j;
  double        **aXu, dx, dy, x, y;
  ierr = VecSet(Xu,PETSC_NINFINITY);CHKERRQ(ierr);
  ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  dx = 1.0 / (PetscReal)(info.mx-1);
  dy = 1.0 / (PetscReal)(info.my-1);
  ierr = DMDAVecGetArray(da, Xu, &aXu);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * dx;
      aXu[j][i] = psi(x,y);
    }
  }
  ierr = DMDAVecRestoreArray(da, Xu, &aXu);CHKERRQ(ierr);
  return 0;
}


