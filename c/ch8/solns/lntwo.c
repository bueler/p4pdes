static char help[] = "Compute ln 2 in serial with PETSc, using random\n"
"permutation of sum order.  Shows that floating-point arithmetic is not\n"
"associative.\n\n";

#include <petsc.h>
#include <time.h>

int main(int argc, char **args) {
  PetscErrorCode  ierr;
  PetscMPIInt     size;
  PetscInt        i, j, n=10;
  PetscReal       v, tmp, *a, sum;
  PetscRandom     r;

  PetscInitialize(&argc,&args,NULL,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
  if (size > 1) {
      SETERRQ(PETSC_COMM_WORLD,1,"lntwo only works in serial\n");
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for lntwo",""); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-n","number of terms in sum",
                          "lntwo.c",n,&n,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r); CHKERRQ(ierr);
  ierr = PetscRandomSetType(r,PETSCRAND48); CHKERRQ(ierr);
  ierr = PetscRandomSetSeed(r,(int)time(NULL)); CHKERRQ(ierr);
  ierr = PetscRandomSeed(r); CHKERRQ(ierr);

  // fill array with the terms   (-1)^i / (i+1)  for i=0 .. n
  ierr = PetscMalloc1(n,&a); CHKERRQ(ierr);
  for (i=0; i<n; i++) {
      a[i] = pow(-1.0,(double)i) / (double)(i+1);
  }

  // shuffle the terms
  for (i=n-1; i>0; i--) {
      ierr = PetscRandomGetValueReal(r,&v); CHKERRQ(ierr);
      j = (int)floor(i*v);
      tmp = a[i];
      a[i] = a[j];
      a[j] = tmp;
  }

  // sum the terms
  sum = 0.0;
  for (i=0; i<n; i++) {
      sum += a[i];
  }

  // print the result
  ierr = PetscPrintf(PETSC_COMM_WORLD,"ln 2 is approximately %18.16f\n",sum); CHKERRQ(ierr);

  PetscRandomDestroy(&r);
  PetscFree(a);
  return PetscFinalize();
}
