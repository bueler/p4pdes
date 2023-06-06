static char help[] = "Backward Euler time stepping using DMDA+KSP.\n"
"This code is the minimal modification of poisson.c to do time-stepping.\n"
"We solve\n"
"  u_t = Laplacian u + f(x,y)\n"
"with homogeneous Dirichlet boundary conditions.  Initial condition is\n"
"u(0,x,y)=0.  Of course, recommended approaches are to use TS (Chapter 5)\n"
"or to solve each time step problem using the SNES (Chapters 4 and etc).\n"
"Example:\n"
"  $ ./heatsteps -da_refine 4 -ksp_view_solution draw\n\n";

#include <petsc.h>

extern PetscErrorCode formMatrix(DM, Mat);
extern PetscErrorCode formExact(DM, Vec);
extern PetscErrorCode formRHS(DM, Vec, Vec);

// BAD design: the time step and number of steps should not be global constants
// (use application context)
PetscReal deltat = 0.01;
PetscInt  NSTEPS = 50;

//STARTMAIN
int main(int argc,char **args) {
    DM            da;
    Mat           A;
    Vec           b,u,uexact,
                  uold;  // added for time-stepping
    KSP           ksp;
    PetscReal     errnorm;
    DMDALocalInfo info;
    PetscInt      k;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    // change default 9x9 size using -da_grid_x M -da_grid_y N
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                 DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                 9,9,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));

    // create linear system matrix A
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMCreateMatrix(da,&A));
    PetscCall(MatSetFromOptions(A));

    // create RHS b, approx solution u, exact solution uexact
    PetscCall(DMCreateGlobalVector(da,&b));
    PetscCall(VecDuplicate(b,&u));
    PetscCall(VecDuplicate(b,&uexact));

    // added for time-stepping
    PetscCall(VecDuplicate(b,&uold));
    PetscCall(VecSet(uold,0.0));  // initial condition

    // fill vectors and assemble linear system
    PetscCall(formExact(da,uexact));
    //PetscCall(formRHS(da,b));
    PetscCall(formMatrix(da,A));

    // create and solve the linear system
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));

    // time-stepping loop; THE BIG CHANGE IS HERE
    for (k=0; k<NSTEPS; k++) {
       PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                 "  solving for u_%2d = u(%.3f,x,y) ...\n",k+1,(k+1)*deltat));
       PetscCall(formRHS(da,uold,b));
       PetscCall(KSPSolve(ksp,b,u));
       PetscCall(VecCopy(u,uold));
    }

    // report on grid and numerical error
    PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uxact
    PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
    PetscCall(DMDAGetLocalInfo(da,&info));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                //"on %d x %d grid:  error |u-uexact|_inf = %g\n",
                "on %d x %d grid:  error relative to steady state |u-uexact|_inf = %g\n",
                info.mx,info.my,errnorm));

    PetscCall(VecDestroy(&uold));
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
// Note: this sets up the matrix for backward Euler.  If A is the original
// matrix for -Laplacian, but scaled as explained in Chapter 3, so A is SPD
// with O(1) entries, then this is forming
//   A_new = I + (dt/(hx*hy)) A
PetscErrorCode formMatrix(DM da, Mat A) {
    DMDALocalInfo  info;
    MatStencil     row, col[5];
    PetscReal      hx, hy, v[5];
    PetscInt       i, j, ncols;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    for (j = info.ys; j < info.ys+info.ym; j++) {
        for (i = info.xs; i < info.xs+info.xm; i++) {
            row.j = j;           // row of A corresponding to (x_i,y_j)
            row.i = i;
            col[0].j = j;        // diagonal entry
            col[0].i = i;
            ncols = 1;
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                v[0] = 1.0;      // on boundary: trivial equation
            } else {
                //v[0] = 2*(hy/hx + hx/hy); // interior: build a row
                v[0] = 1.0 + deltat * 2*(1.0/(hx*hx) + 1.0/(hy*hy));
                if (i-1 > 0) {
                    col[ncols].j = j;    col[ncols].i = i-1;
                    //v[ncols++] = -hy/hx;
                    v[ncols++] = -deltat/(hx*hx);
                }
                if (i+1 < info.mx-1) {
                    col[ncols].j = j;    col[ncols].i = i+1;
                    //v[ncols++] = -hy/hx;
                    v[ncols++] = -deltat/(hx*hx);
                }
                if (j-1 > 0) {
                    col[ncols].j = j-1;  col[ncols].i = i;
                    //v[ncols++] = -hx/hy;
                    v[ncols++] = -deltat/(hy*hy);
                }
                if (j+1 < info.my-1) {
                    col[ncols].j = j+1;  col[ncols].i = i;
                    //v[ncols++] = -hx/hy;
                    v[ncols++] = -deltat/(hy*hy);
                }
            }
            PetscCall(MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES));
        }
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    return 0;
}
//ENDMATRIX

//STARTEXACT
// NOTE: this is the exact solution to the Poisson equation -Laplacian u = f,
// so it is the exact steady state only.
PetscErrorCode formExact(DM da, Vec uexact) {
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, **auexact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
    for (j = info.ys; j < info.ys+info.ym; j++) {
        y = j * hy;
        for (i = info.xs; i < info.xs+info.xm; i++) {
            x = i * hx;
            auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}

// Modified RHS to use previous timestep uold.
PetscErrorCode formRHS(DM da, Vec uold, Vec b) {
    PetscInt       i, j;
    PetscReal      hx, hy, x, y, f, **ab;
    const PetscReal **auold;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(da,&info));
    hx = 1.0/(info.mx-1);  hy = 1.0/(info.my-1);
    PetscCall(DMDAVecGetArray(da, b, &ab));
    PetscCall(DMDAVecGetArrayRead(da, uold, &auold));
    for (j=info.ys; j<info.ys+info.ym; j++) {
        y = j * hy;
        for (i=info.xs; i<info.xs+info.xm; i++) {
            x = i * hx;
            if (i==0 || i==info.mx-1 || j==0 || j==info.my-1) {
                ab[j][i] = 0.0;  // on boundary: 1*u = 0
            } else {
                f = 2.0 * ( (1.0 - 6.0*x*x) * y*y * (1.0 - y*y)
                    + (1.0 - 6.0*y*y) * x*x * (1.0 - x*x) );
                //ab[j][i] = hx * hy * f;
                ab[j][i] = auold[j][i] + deltat * f;  // for time-stepping
            }
        }
    }
    PetscCall(DMDAVecRestoreArrayRead(da, uold, &auold));
    PetscCall(DMDAVecRestoreArray(da, b, &ab));
    return 0;
}
//ENDEXACT
