
static char help[] = "Solves a 1D Poisson problem with DMDA and KSP.\n\n";

#include <petsc.h>

//FIXME
PetscErrorCode formdirichletlaplacian(DM da, PetscReal diagentry, Mat A) {
    PetscErrorCode ierr;
    DMDALocalInfo  info;
    PetscInt       i;

    ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[3];
      PetscReal   v[3];
      PetscInt    ncols = 0;
      row.i = i;
      col[ncols].i = i;
      if ( (i==0) || (i==info.mx-1) ) {
        // if on boundary, just insert diagonal entry
        v[ncols++] = diagentry;
      } else {
        v[ncols++] = 2; // ... everywhere else we build a row
        // if neighbor is NOT a known boundary value then we put an entry:
        if (i-1>0) {
          col[ncols].i = i-1;  v[ncols++] = -1;  }
        if (i+1<info.mx-1) {
          col[ncols].i = i+1;  v[ncols++] = -1;  }
      }
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode formExact(DM da, Vec uexact) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx, x, *auexact;

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  hx = 1.0/(info.mx-1);
  ierr = DMDAVecGetArray(da, uexact, &auexact);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    x = i * hx;
    auexact[i] = x*x * (1.0 - x*x);
  }
  ierr = DMDAVecRestoreArray(da, uexact, &auexact);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode formRHS(DM da, Vec b) {
  PetscErrorCode ierr;
  PetscInt       i;
  PetscReal      hx, x, x2, f, *ab;
  DMDALocalInfo  info;

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  hx = 1.0/(info.mx-1);
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    x = i * hx;  x2 = x*x;
    if ( (i>0) && (i<info.mx-1) ) { // if not bdry
      // f = - (u_xx + u_yy)  where u is exact
      f = 2.0 * (1.0 - 6.0*x2);  //FIXME
      ab[i] = hx * f;
    } else {
      ab[i] = 0.0;                          // on bdry we have 1*u = 0
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &ab); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  return 0;
}


int main(int argc,char **args) {
  PetscErrorCode ierr;
  PetscInitialize(&argc,&args,(char*)0,help);

  DM  da;
//DMDACreate1d(MPI Comm comm,DMDABoundaryType xperiod,int M,int w,int s,int *lc,DM *inra);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,
               -9,1,1,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);

  // create linear system matrix A
  Mat  A;
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);

  // create right-hand-side (RHS) b, approx solution u, exact solution uexact
  Vec  b,u,uexact;
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);

  // fill known vectors
  ierr = formExact(da,uexact); CHKERRQ(ierr);
  ierr = formRHS(da,b); CHKERRQ(ierr);

  // assemble linear system
  PetscLogStage  stage; //STRIP
  ierr = PetscLogStageRegister("Matrix Assembly", &stage); CHKERRQ(ierr); //STRIP
  ierr = PetscLogStagePush(stage); CHKERRQ(ierr); //STRIP
  ierr = formdirichletlaplacian(da,1.0,A); CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr); //STRIP

  // create linear solver context
  KSP  ksp;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // solve
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);

  // report on grid and numerical error
  PetscScalar    errnorm;
  DMDALocalInfo  info;
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d point grid:  error |u-uexact|_inf = %g\n",
             info.mx,errnorm); CHKERRQ(ierr);

  KSPDestroy(&ksp);
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  MatDestroy(&A);
  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

