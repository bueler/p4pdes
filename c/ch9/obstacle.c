static const char help[] = "Solves obstacle problem in 2D as a variational inequality.\n\
An elliptic problem with solution  u  constrained to be above a given function  psi. \n\
Exact solution is known.  Because of the constraint, the problem is nonlinear.\n";

/*
Parallel runs, spatial refinement only, robust PC:
  for M in 0 1 2 3 4 5 6; do
    mpiexec -n 4 ./obstacle -da_refine $M -snes_converged_reason -pc_type asm -sub_pc_type lu
  done
*/

#include <petsc.h>

/* application context for obstacle problem solver */
typedef struct {
  PetscReal dx, dy;
  Vec psi,  // obstacle
      g;    // Dirichlet boundary conditions
} ObsCtx;

PetscErrorCode FormDirichletPsiExactInitial(DM da,Vec Uexact,Vec U0,PetscBool feasibleU0, ObsCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      **ag, **apsi, **au0, **auexact,
                 x, y, r,
                 pi = PETSC_PI, afree = 0.69797, A = 0.68026, B = 0.47152;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, user->g, &ag);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, user->psi, &apsi);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, U0, &au0);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = -2.0 + j * user->dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = -2.0 + i * user->dx;
      r = PetscSqrtReal(x * x + y * y);
      if (r <= 1.0)
        apsi[j][i] = PetscSqrtReal(1.0 - r * r);
      else
        apsi[j][i] = -1.0;
      if (r <= afree)
        auexact[j][i] = apsi[j][i];  /* on the obstacle */
      else
        auexact[j][i] = - A * PetscLogReal(r) + B;   /* solves the laplace eqn */
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1)
        ag[j][i] = auexact[j][i];
      else
        ag[j][i] = NAN;
      if (feasibleU0) {
        au0[j][i] = auexact[j][i] + PetscCosReal(pi*x/4.0)*PetscCosReal(pi*y/4.0);
      } else
        au0[j][i] = 0.;
    }
  }
  ierr = DMDAVecRestoreArray(da, user->g, &ag);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, user->psi, &apsi);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, U0, &au0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//  for call-back: tell SNESVI (variational inequality) that we want  psi <= u < +infinity
PetscErrorCode FormBounds(SNES snes, Vec Xl, Vec Xu)
{
  PetscErrorCode ierr;
  ObsCtx         *user;

  PetscFunctionBeginUser;
  ierr = SNESGetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = VecCopy(user->psi,Xl);CHKERRQ(ierr);  /* u >= psi */
  ierr = VecSet(Xu,PETSC_INFINITY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,PetscScalar **x,PetscScalar **f,ObsCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      uxx,uyy,**ag;

  PetscFunctionBeginUser;
  ierr = DMDAVecGetArray(info->da, user->g, &ag);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i] - ag[j][i];
      } else {
        uxx     = (x[j][i-1] - 2.0 * x[j][i] + x[j][i+1]) / (user->dx*user->dx);
        uyy     = (x[j-1][i] - 2.0 * x[j][i] + x[j+1][i]) / (user->dy*user->dy);
        f[j][i] = - uxx - uyy;
      }
    }
  }
  ierr = DMDAVecRestoreArray(info->da, user->g, &ag);CHKERRQ(ierr);

  ierr = PetscLogFlops(10.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* FormJacobianLocal - Evaluates Jacobian matrix on local process patch */
// FIXME can be formed symmetrically
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info,PetscScalar **x,Mat A,Mat jac, ObsCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  MatStencil     col[5],row;
  PetscReal      v[5],oxx,oyy;

  PetscFunctionBeginUser;
  oxx = 1.0 / (user->dx * user->dx);
  oyy = 1.0 / (user->dy * user->dy);

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      row.j = j; row.i = i;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) { /* boundary */
        v[0] = 1.0;
        ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
      } else { /* interior grid points */
        v[0] = -oyy;                 col[0].j = j - 1;  col[0].i = i;
        v[1] = -oxx;                 col[1].j = j;      col[1].i = i - 1;
        v[2] = 2.0 * (oxx + oyy);    col[2].j = j;      col[2].i = i;
        v[3] = -oxx;                 col[3].j = j;      col[3].i = i + 1;
        v[4] = -oyy;                 col[4].j = j + 1;  col[4].i = i;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  /* Assemble matrix, using the 2-step process: */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != jac) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  /* Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error. */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode      ierr;
  SNES                snes;
  Vec                 u, r, uexact;   /* solution, residual vector, exact solution */
  DM                  da;
  ObsCtx              user;
  PetscReal           error1,errorinf;
  PetscBool           feasible = PETSC_FALSE,fdflg = PETSC_FALSE;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,NULL,help);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      5,5,                       // override with -da_grid_x,_y or -da_refine
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da); CHKERRQ(ierr);
  ierr = DMSetUp(da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,-1.0,-1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  user.dx = 4.0 / (PetscReal)(info.mx-1);
  user.dy = 4.0 / (PetscReal)(info.my-1);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&uexact);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.g));CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.psi));CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options to obstacle problem","");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fd","use coloring to compute Jacobian by finite differences",
                    NULL,fdflg,&fdflg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-feasible","use feasible initial guess",
                    NULL,feasible,&feasible,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = FormDirichletPsiExactInitial(da,uexact,u,feasible,&user);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
  ierr = SNESSetApplicationContext(snes,&user);CHKERRQ(ierr);
  ierr = SNESSetType(snes,SNESVINEWTONRSLS);CHKERRQ(ierr);
  ierr = SNESVISetComputeVariableBounds(snes,&FormBounds);CHKERRQ(ierr);
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,
            (PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,
            &user);CHKERRQ(ierr);
  if (!fdflg) {
    ierr = DMDASNESSetJacobianLocal(da,
              (PetscErrorCode (*)(DMDALocalInfo*,void*,Mat,Mat,void*))FormJacobianLocal,
              &user);CHKERRQ(ierr);
  }
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* report on setup */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"setup done on %D x %D grid\n",
                     info.mx, info.my);CHKERRQ(ierr);

  /* solve nonlinear system */
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);

  /* compare to exact */
  ierr = VecWAXPY(r,-1.0,uexact,u);CHKERRQ(ierr); /* r = u - uexact */
  ierr = VecNorm(r,NORM_1,&error1);CHKERRQ(ierr);
  error1 /= (PetscReal)info.mx * (PetscReal)info.my;
  ierr = VecNorm(r,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
               "errors:    av |u-uexact|  = %.3e\n"
               "           |u-uexact|_inf = %.3e\n",
               (double)error1,(double)errorinf);CHKERRQ(ierr);

  VecDestroy(&u); VecDestroy(&r); VecDestroy(&uexact);
  VecDestroy(&(user.psi)); VecDestroy(&(user.g));
  SNESDestroy(&snes); DMDestroy(&da);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

