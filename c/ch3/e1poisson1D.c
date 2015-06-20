
static char help[] = "Solves a 1D Poisson problem with DMDA and KSP.\n\n";

#include <petsc.h>

PetscErrorCode formdirichletlaplacian(DM da, Mat A) {
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
        v[ncols++] = 1;
      } else {
        v[ncols++] = 2;
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

PetscErrorCode formExactAndRHS(DM da, Vec uexact, Vec b) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      hx, x, x2, *ab, *auexact;

  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);
  hx = 1.0/(info.mx-1);
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, uexact, &auexact);CHKERRQ(ierr);
  for (i=info.xs; i<info.xs+info.xm; i++) {
    x = i * hx;  x2 = x*x;
    auexact[i] = x2 * (1.0 - x2);
    if ( (i>0) && (i<info.mx-1) ) { // if not bdry
      ab[i] = hx * 2.0 * (1.0 - 6.0*x2);  //FIXME
    } else {
      ab[i] = 0.0;
    }
  }
  ierr = DMDAVecRestoreArray(da, uexact, &auexact);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, b, &ab); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **args) {
  PetscErrorCode ierr;
  DM             da;
  KSP            ksp;
  Mat            A;
  Vec            b,u,uexact;
  PetscReal      errnorm;
  DMDALocalInfo  info;
  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,
               -9,1,1,NULL,
               &da); CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0); CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = formExactAndRHS(da,uexact,b); CHKERRQ(ierr);
  ierr = formdirichletlaplacian(da,A); CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);    // u <- u + (-1.0) uxact
  ierr = VecNorm(u,NORM_INFINITY,&errnorm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,
             "on %d point grid:  error |u-uexact|_inf = %g\n",
             info.mx,errnorm); CHKERRQ(ierr);
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  MatDestroy(&A);  KSPDestroy(&ksp);  DMDestroy(&da);
  PetscFinalize();
  return 0;
}

