
static char help[] = "SSolves a structured-grid Poisson problem with DMDA and KSP,\n"
"but (unlike c2poisson) also using KSPSetComputeOperators() so we can use\n"
"multigrid preconditioning at the command line.\n\n";

// based on src/ksp/ksp/examples/tutorials/ex50.c, but in Dirichlet-only case,
//   and following c2poisson.c closely

// FIXME:currently code duplication from c2poisson.c; is there a good way to do code re-use on stuff from structuredlaplacian.c?

// USE MULTIGRID AND SHOW IT GRAPHICALLY:
//   ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 3 -ksp_monitor -ksp_view -dm_view draw -draw_pause 1

// NOT SURE WHAT IS BEING ACCOMPLISHED HERE:
//   ./c4poisson -da_grid_x 100 -da_grid_y 100 -pc_type mg  -pc_mg_levels 1 -mg_levels_0_pc_type ilu -mg_levels_0_pc_factor_levels 1 -ksp_monitor -ksp_view
//   ./c4poisson -da_grid_x 100 -da_grid_y 100 -pc_type mg -pc_mg_levels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor

// I NEED TO KNOW/UNDERSTAND EVERYTHING IN foo.txt:
//   mpiexec -n 4 ./c4poisson -da_grid_x 3 -da_grid_y 3 -pc_type mg -da_refine 10 -ksp_monitor -dm_view -ksp_view -log_summary &> foo.txt

#include <math.h>
#include <petscdmda.h>
#include <petscksp.h>


//COMPUTERHS
PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  DM             da;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  PetscInt       i, j;
  PetscReal      hx = 1./(double)(info.mx-1),
                 hy = 1./(double)(info.my-1),  // domain is [0,1] x [0,1]
                 pi = PETSC_PI, x, y, f, **ab, uex;
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * hy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * hx;
      // choose exact solution to satisfy boundary conditions, and be a bit
      //   generic (e.g. not equal to an eigenvector)
      uex = x * (1.0 - x) * sin(3.0 * pi * y);
      if ( (i>0) && (i<info.mx-1) && (j>0) && (j<info.my-1) ) { // if not bdry
        // f = - (u_xx + u_yy)
        f = 2 * sin(3.0 * pi * y) + 9.0 * pi * pi * uex;
        ab[j][i] = hx * hy * f;
      } else {
        ab[j][i] = 0.0;                          // on bdry we have "1 * u = 0"
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &ab);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  return 0;
}
//ENDCOMPUTERHS


//COMPUTEJAC
PetscErrorCode ComputeJacobian(KSP ksp,Mat J, Mat A,void *ctx) {
  PetscErrorCode ierr;
  PetscInt       i, j;
  PetscScalar    hx, hy;
  DM             da;
  DMDALocalInfo  info;

  PetscFunctionBeginUser;
  ierr  = KSPGetDM(ksp,&da);CHKERRQ(ierr);

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx);
  hy   = 1.0/(PetscReal)(info.my);

  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscReal   v[5];
      PetscInt    ncols = 0;
      row.j = j;               // row of A corresponding to the unknown at (x_i,y_j)
      row.i = i;
      col[ncols].j = j;        // in that diagonal entry ...
      col[ncols].i = i;
      if ( (i==0) || (i==info.mx-1) || (j==0) || (j==info.my-1) ) { // ... on bdry
        v[ncols++] = 1.0;
      } else {
        v[ncols++] = 2*(hy/hx + hx/hy); // ... everywhere else we build a row
        // if neighbor is NOT known to be zero we put an entry:
        if (i-1>0) {
          col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = -hy/hx;  }
        if (i+1<info.mx-1) {
          col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = -hy/hx;  }
        if (j-1>0) {
          col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
        if (j+1<info.my-1) {
          col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
      }
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//ENDCOMPUTEJAC

//MAIN
int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,
                DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                DMDA_STENCIL_STAR,-10,-10,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,
                &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,-1.0,-1.0); CHKERRQ(ierr);

  ierr = KSPSetDM(ksp,(DM)da);

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeJacobian,NULL);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
//ENDMAIN

