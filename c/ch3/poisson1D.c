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

    // change default 9x9 size using -da_grid_x M
    /* PetscCall(DMDACreate2d(PETSC_COMM_WORLD, */
    /*              DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, */
    /*              9,9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da)); */
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD,
                 DM_BOUNDARY_NONE, 9,1,1,NULL,&da));

    
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
                "on %d grid:  error |u-uexact|_inf = %g\n",
                info.mx,errnorm));

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
    PetscInt       i=0;

    PetscCall(DMDAGetLocalInfo(da,&info));
    
    for (i = info.xs; i < info.xs+info.xm; i++) {
      PetscInt     row = i, col[3];
      PetscReal      v[3];
      PetscInt       ncols=0;
      
      if (i==0 || i==info.mx-1) {
	col[ncols] = i; v[ncols++] = 1.0;
      } else {
	col[ncols] = i; v[ncols++]=2.0;
	if (i-1 > 0) {
	  col[ncols] = i-1;
	  v[ncols++] = -1.0;
	}
	if (i+1 < info.mx-1) {
	  col[ncols] = i+1;
	  v[ncols++] = -1.0;
	}
      }
      PetscCall(MatSetValues(A ,1,&row,ncols,col,v,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    return 0;
}
//ENDMATRIX

//STARTEXACT
PetscErrorCode formExact(DM da, Vec uexact) {
    PetscInt       i;
    PetscReal      hx, x, *auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
    for (i = info.xs; i < info.xs+info.xm; i++) {
        x = i * hx;
        auexact[i] = x*x * (1.0 - x*x);
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}

PetscErrorCode formRHS(DM da, Vec b) {
    PetscInt       i;
    PetscReal      hx, x, f, *ab;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);
    PetscCall(DMDAVecGetArray(da, b, &ab));
    for (i=info.xs; i<info.xs+info.xm; i++) {
        x = i * hx;
        if (i==0 || i==info.mx-1) {
            ab[i] = 0.0;  // on boundary: 1*u = 0
        } else {
	  f = 2.0 -12 * x * x;
	  ab[i] = -hx * hx * f;
        }
    }
    PetscCall(DMDAVecRestoreArray(da, b, &ab));
    return 0;
}
//ENDEXACT

