static char help[] = "Solves a 1D Poisson problem with DMDA and KSP.\n\n";

#include <petsc.h>

extern PetscErrorCode formdirichletlaplacian(DM, Mat);
extern PetscErrorCode formExactAndRHS(DM, Vec, Vec);

int main(int argc,char **args) {
  DM             da;
  KSP            ksp;
  Mat            A;
  Vec            b,u,uexact;
  PetscReal      errnorm;
  DMDALocalInfo  info;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,
                         9,1,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,-1.0,-1.0,-1.0,-1.0));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatSetOptionsPrefix(A,"a_"));
  PetscCall(MatSetFromOptions(A));
  PetscCall(DMCreateGlobalVector(da,&b));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecDuplicate(b,&uexact));
  PetscCall(formExactAndRHS(da,uexact,b));
  PetscCall(formdirichletlaplacian(da,A));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,u));
  PetscCall(VecAXPY(u,-1.0,uexact));    // u <- u + (-1.0) uxact
  PetscCall(VecNorm(u,NORM_INFINITY,&errnorm));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "on %d point grid:  error |u-uexact|_inf = %g\n",
                        info.mx,errnorm));
  VecDestroy(&u);  VecDestroy(&uexact);  VecDestroy(&b);
  MatDestroy(&A);  KSPDestroy(&ksp);  DMDestroy(&da);
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode formdirichletlaplacian(DM da, Mat A) {
    DMDALocalInfo  info;
    PetscInt       i;

    PetscCall(DMDAGetLocalInfo(da,&info));
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscReal   v[3];
      PetscInt    row = i, col[3];
      PetscInt    ncols=0;
      if ( (i==0) || (i==info.mx-1) ) {
        col[ncols] = i;  v[ncols++] = 1.0;
      } else {
        col[ncols] = i;  v[ncols++] = 2.0;
        if (i-1>0) {
          col[ncols] = i-1;  v[ncols++] = -1.0;  }
        if (i+1<info.mx-1) {
          col[ncols] = i+1;  v[ncols++] = -1.0;  }
      }
      PetscCall(MatSetValues(A,1,&row,ncols,col,v,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    return 0;
}

PetscErrorCode formExactAndRHS(DM da, Vec uexact, Vec b) {
  DMDALocalInfo  info;
  PetscInt       i;
  PetscReal      h, x, *ab, *auexact;

  PetscCall(DMDAGetLocalInfo(da,&info));
  h = 1.0/(info.mx-1);
  PetscCall(DMDAVecGetArray(da, b, &ab));
  PetscCall(DMDAVecGetArray(da, uexact, &auexact));
  for (i=info.xs; i<info.xs+info.xm; i++) {
    x = i * h;
    auexact[i] = x*x * (1.0 - x*x);
    if ( (i>0) && (i<info.mx-1) )
      ab[i] = h*h * (12.0 * x*x - 2.0);
    else
      ab[i] = 0.0;
  }
  PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
  PetscCall(DMDAVecRestoreArray(da, b, &ab));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecAssemblyBegin(uexact));
  PetscCall(VecAssemblyEnd(uexact));
  return 0;
}
