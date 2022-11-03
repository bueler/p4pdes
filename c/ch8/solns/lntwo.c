static char help[] = "Compute ln 2 in serial with PETSc, using random\n"
"permutation of sum order.  Shows that floating-point arithmetic is not\n"
"associative.\n\n";

#include <petsc.h>
#include <time.h>

int main(int argc, char **args) {
  PetscMPIInt     size;
  PetscInt        i, j, n=10;
  PetscReal       v, tmp, *a, sum;
  PetscRandom     r;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));

  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  if (size > 1) {
      SETERRQ(PETSC_COMM_WORLD,1,"lntwo only works in serial\n");
  }

  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for lntwo","");
  PetscCall(PetscOptionsInt("-n","number of terms in sum",
                            "lntwo.c",n,&n,NULL));
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&r));
  PetscCall(PetscRandomSetType(r,PETSCRAND48));
  PetscCall(PetscRandomSetSeed(r,(int)time(NULL)));
  PetscCall(PetscRandomSeed(r));

  // fill array with the terms   (-1)^i / (i+1)  for i=0 .. n
  PetscCall(PetscMalloc1(n,&a));
  for (i=0; i<n; i++) {
      a[i] = pow(-1.0,(double)i) / (double)(i+1);
  }

  // shuffle the terms
  for (i=n-1; i>0; i--) {
      PetscCall(PetscRandomGetValueReal(r,&v));
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
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"ln 2 is approximately %18.16f\n",sum));

  PetscCall(PetscRandomDestroy(&r));
  PetscCall(PetscFree(a));
  PetscCall(PetscFinalize());
  return 0;
}
