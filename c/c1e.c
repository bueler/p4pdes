static char help[] = "To get started with PETSc, compute e in parallel.\n\n";

#include <petscsys.h>

int main(int argc,char **args) {
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  PetscScalar     localval, globalsum;
  PetscLogDouble  flops;
  int             i;

  PetscInitialize(&argc,&args,(char*)0,help);  // <-- always call

  // get my rank
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  // compute  1 / n!  where n = (one more than rank of process)
  localval = 1.0;
  for (i = 1; i < rank; i++)
    localval *= i+1;
  localval = 1.0 / localval;
  // record the flops we just did, for later reporting
  if (rank > 0) {
    ierr = PetscLogFlops(2 * (rank - 1) + 1); CHKERRQ(ierr);
  }

  // sum the contributions over all processes
  ierr = MPI_Allreduce(&localval, &globalsum, 1, MPI_DOUBLE, MPI_SUM,
                       PETSC_COMM_WORLD); CHKERRQ(ierr);

  // output is estimate of e, and report on work done
  ierr = PetscPrintf(PETSC_COMM_WORLD,"e is about %17.15f\n",globalsum); CHKERRQ(ierr);
  ierr = PetscGetFlops(&flops); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"rank %d did %d flops\n",rank,(int)flops); CHKERRQ(ierr);

  PetscFinalize();  // <-- always call
  return 0;
}
