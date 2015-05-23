static char help[] = "Compute e in parallel with PETSc.\n\n";

#include <petsc.h>

int main(int argc, char **args) {
  PetscErrorCode  ierr;
  PetscMPIInt     rank;
  PetscScalar     localval, globalsum;
  int             i;

  PetscInitialize(&argc,&args,(char*)0,help);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  // compute  1 / n!  where n = (rank of process) + 1
  localval = 1.0;
  for (i = 2; i < rank+1; i++)
    localval /= i;

  // sum the contributions over all processes
  ierr = MPI_Allreduce(&localval, &globalsum, 1, MPI_DOUBLE, MPI_SUM,
                       PETSC_COMM_WORLD); CHKERRQ(ierr);

  // output one estimate of e, but report on work from each process
  ierr = PetscPrintf(PETSC_COMM_WORLD,
    "e is about %17.15f\n",globalsum); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,
    "rank %d did %d flops\n",rank,(rank > 0) ? rank-1 : 0); CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
