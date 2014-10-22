static const char help[] = "Solves obstacle problem in 2D as a variational inequality.\n\
An elliptic problem with solution  u  constrained to be above a given function  psi. \n\
Exact solution is known.  Because of the constraint, the problem is nonlinear.\n";

/*
Example usage follows.

Get help:
  ./ex9 -help

Parallel runs, spatial refinement only:
  for M in 21 41 81 161 321; do
    echo "case M=$M:"
    mpiexec -n 4 ./ex9 -da_grid_x $M -da_grid_y $M -snes_monitor
  done

With finite difference evaluation of Jacobian using coloring:
  ./ex9 -fd

*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

//STRUCT
/* application context for obstacle problem solver */
typedef struct {
  PetscReal dx, dy;
  Vec psi,  // obstacle
      g;    // Dirichlet boundary conditions
} ObsCtx;
//ENDSTRUCT


//FORMPSI
PetscErrorCode FormPsiAndInitialGuess(DM da,Vec U0,PetscBool feasible, ObsCtx *user)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      **psi, **u0, **uexact,
                 x, y, r,
                 pi = PETSC_PI, afree = 0.69797, A = 0.68026, B = 0.47152;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, user->psi, &psi);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, U0, &u0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, user->g, &uexact);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = -2.0 + j * user->dy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = -2.0 + i * user->dx;
      r = PetscSqrtReal(x * x + y * y);
      if (r <= 1.0)
        psi[j][i] = PetscSqrtReal(1.0 - r * r);
      else
        psi[j][i] = -1.0;
      if (r <= afree)
        uexact[j][i] = psi[j][i];  /* on the obstacle */
      else
        uexact[j][i] = - A * PetscLogReal(r) + B;   /* solves the laplace eqn */
      if (feasible) {
        if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1)
           u0[j][i] = uexact[j][i];
        else
           u0[j][i] = uexact[j][i] + PetscCosReal(pi*x/4.0)*PetscCosReal(pi*y/4.0);
      } else
        u0[j][i] = 0.;
    }
  }
  ierr = DMDAVecRestoreArray(da, user->psi, &psi);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, U0, &u0);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, user->g, &uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//ENDFORMPSI


//FORMBOUNDS
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
//ENDFORMBOUNDS


/* FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch */
//FORMFUNC
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
//ENDFORMFUNC


/* FormJacobianLocal - Evaluates Jacobian matrix on local process patch */
//FORMJAC
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
//ENDFORMJAC

int main(int argc,char **argv)
{
  PetscErrorCode      ierr;
  SNES                snes;
  Vec                 u, r;   /* solution, residual vector */
  PetscInt            its;
  SNESConvergedReason reason;
  DM                  da;
  ObsCtx              user;
  PetscReal           error1,errorinf;
  PetscBool           feasible = PETSC_FALSE,fdflg = PETSC_FALSE;
  DMDALocalInfo  info;

  PetscInitialize(&argc,&argv,(char*)0,help);

//CREATE
  ierr = DMDACreate2d(PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
      -11,-11,                   // override with -da_grid_x,_y or -da_refine
      PETSC_DECIDE,PETSC_DECIDE, // num of procs in each dim
      1,1,NULL,NULL,             // dof = 1 and stencil width = 1
      &da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,-2.0,2.0,-2.0,2.0,-1.0,-1.0);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  user.dx = 4.0 / (PetscReal)(info.mx-1);
  user.dy = 4.0 / (PetscReal)(info.my-1);

  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.g));CHKERRQ(ierr);
  ierr = VecDuplicate(u,&(user.psi));CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options to obstacle problem","");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-fd","use coloring to compute Jacobian by finite differences",
                    NULL,fdflg,&fdflg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-feasible","use feasible initial guess",
                    NULL,feasible,&feasible,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
//ENDCREATE

//SETUPSNES
  ierr = FormPsiAndInitialGuess(da,u,feasible,&user);CHKERRQ(ierr);
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
//ENDSETUPSNES

  /* report on setup */
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "setup done: square       side length = %.3f\n"
                     "            grid               Mx,My = %D,%D\n"
                     "            spacing            dx,dy = %.3f,%.3f\n",
                     4.0, info.mx, info.my, user.dx, user.dy);CHKERRQ(ierr);

//SOLVE
  /* solve nonlinear system */
  ierr = SNESSolve(snes,NULL,u);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"number of Newton iterations = %D; result = %s\n",
                     its,SNESConvergedReasons[reason]);CHKERRQ(ierr);

  /* compare to exact */
  ierr = VecWAXPY(r,-1.0,user.g,u);CHKERRQ(ierr); /* r = u - g = u - uexact */
  ierr = VecNorm(r,NORM_1,&error1);CHKERRQ(ierr);
  error1 /= (PetscReal)info.mx * (PetscReal)info.my;
  ierr = VecNorm(r,NORM_INFINITY,&errorinf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
               "errors:    av |u-uexact|  = %.3e\n"
               "           |u-uexact|_inf = %.3e\n",
               (double)error1,(double)errorinf);CHKERRQ(ierr);
//ENDSOLVE

  /* Free work space.  */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&(user.psi));CHKERRQ(ierr);
  ierr = VecDestroy(&(user.g));CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

