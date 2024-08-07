static char help[] = "A structured-grid Poisson solver using DMDA+KSP.\n\n";

#include <petsc.h>

extern PetscErrorCode formMatrix(DM, Mat);
extern PetscErrorCode formExact(DM, Vec);
extern PetscErrorCode formRHS(DM, Vec);

//STARTMAIN
int main(int argc,char **args) {
    DM            da;
    Mat           A;
    Vec           b,u,uexact;
    KSP           ksp;
    PetscReal     errnorm;
    DMDALocalInfo info;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    // change default 9x9x9 size using -da_grid_x M -da_grid_y N -da_grid_z P
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
			   DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
			   9,9,9,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&da));

    // create linear system matrix A
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMCreateMatrix(da,&A));
    PetscCall(MatSetFromOptions(A));

    // create RHS b, approx solution u, exact solution uexact
    PetscCall(DMCreateGlobalVector(da,&b));
    PetscCall(VecDuplicate(b,&u));
    PetscCall(VecDuplicate(b,&uexact));

    // fill vectors and assemble linear system
    PetscCall(formExact(da,uexact));
    PetscCall(formRHS(da,b));
    PetscCall(formMatrix(da,A));

    // create and solve the linear system
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,b,u));

    // report on grid and numerical error
    PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uxact
    PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                "on %d x %d x %d grid:  error |u-uexact|_inf = %g\n",
			  info.mx,info.my,info.mz,errnorm));

    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&b));
    PetscCall(MatDestroy(&A));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
//ENDMAIN

//STARTMATRIX
PetscErrorCode formMatrix(DM da, Mat A) {
    DMDALocalInfo  info;
    MatStencil     row, col[7];
    PetscReal      hx, hy, hz, v[7];
    PetscInt       i, j, k, ncols;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1); hz = 1.0/(info.mz-1);
    for (k = info.zs; k < info.zs+info.zm; k++) {
      for (j = info.ys; j < info.ys+info.ym; j++) {
        for (i = info.xs; i < info.xs+info.xm; i++) {
	  row.k = k;
	  row.j = j;           // row of A corresponding to (x_i,y_j,z_k)
	  row.i = i;

	  col[0].k = k;
	  col[0].j = j;        // diagonal entry
	  col[0].i = i;

	  ncols = 1;
	  if (i==0 || i==info.mx-1 || j==0 || j==info.my-1 || k==0 || k==info.mz-1) {
	    v[0] = 1.0;      // on boundary: trivial equation
	  } else {
	    v[0] = 2*(1/hx/hx + 1/hy/hy + 1/hz/hz); // interior: build a row
	    if (i-1 > 0) {
	      col[ncols].k = k; col[ncols].j = j;    col[ncols].i = i-1; 
	      v[ncols++] = -1/ hx / hx;
	    }
	    if (i+1 < info.mx-1) {
	      col[ncols].k = k; col[ncols].j = j;    col[ncols].i = i+1;
	      v[ncols++] = -1/ hx / hx;
	    }
	    if (j-1 > 0) {
	      col[ncols].k = k; col[ncols].j = j-1;  col[ncols].i = i;
	      v[ncols++] = -1/ hy / hy;
	    }
	    if (j+1 < info.my-1) {
	      col[ncols].k = k; col[ncols].j = j+1;  col[ncols].i = i;
	      v[ncols++] = -1/ hy / hy;
	    }
	    if (k-1 > 0) {
	      col[ncols].k = k - 1; col[ncols].j = j;  col[ncols].i = i;
	      v[ncols++] = -1/ hz / hz;
	    }
	    if (k+1 < info.mz-1) {
	      col[ncols].k = k + 1; col[ncols].j = j;  col[ncols].i = i;
	      v[ncols++] = -1/ hz / hz;
	    }
	    
	  }
	  /* printf("ncols: %i, i: %i, j: %i, k: %i\n", ncols, i, j, k); */
	  /* printf("ncols: %i\n", ncols); */
	  PetscCall(MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    return 0;
}
//ENDMATRIX

//STARTEXACT
PetscErrorCode formExact(DM da, Vec uexact) {
  PetscInt       i, j, k;
  PetscReal      hx, hy, hz, x, y, z, ***auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1); hz = 1.0 / (info.mz - 1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
    for (k = info.zs; k < info.zs+info.zm; k++) {
      z = k * hz;
      for (j = info.ys; j < info.ys+info.ym; j++) {
        y = j * hy;
        for (i = info.xs; i < info.xs+info.xm; i++) {
	  x = i * hx;
	  auexact[k][j][i] = x*x * (1.0 - x*x) * y*y * (1.0 - y*y) * z*z * (1.0 - z*z);
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}

PetscErrorCode formRHS(DM da, Vec b) {
  PetscInt       i, j, k;
  PetscReal      hx, hy, hz, x, y, z, f, ***ab;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1); hz=1.0/(info.mz-1);
    PetscCall(DMDAVecGetArray(da, b, &ab));
    for (k=info.zs; k<info.zs+info.zm; k++) {
      z = k * hz;
      for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
	  x = i * hx;
	  if (i==0 || i==info.mx-1 || j==0 || j==info.my-1 || k==0 || k==info.mz-1 ) {
	    ab[k][j][i] = 0.0;  // on boundary: 1*u = 0
	  } else {
	    f = - 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y) * z*z * (1.0 - z*z)
			+ (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) * z*z * (1.0 - z*z)
			+ (1.0 - 6.0*z*z) * x*x * (1.0 - x*x) * y*y * (1.0 - y*y)
			);
	    ab[k][j][i] = f;
	  }
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &ab));
    return 0;
}
//ENDEXACT
