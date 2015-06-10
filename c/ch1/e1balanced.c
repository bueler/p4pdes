static char help[] = "Solution to exercise 1 in chapter 1.\n\n";

#include <petscsys.h>

int main(int argc,char **args) {
  PetscErrorCode  ierr;
  PetscMPIInt     rank, size;
  PetscScalar     localval, globalsum;

  PetscInitialize(&argc,&args,(char*)0,help);  // <-- always call

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  // compute  1 / n!  where n = (one more than rank of process)
  // by getting (n-1)! from previous processor
  if (rank == 0) {
    localval = 1.0;
    if (size > 1) {
      ierr = MPI_Send(&localval, 1, MPI_DOUBLE, rank+1, 1,
               PETSC_COMM_WORLD); CHKERRQ(ierr);
    }
  } else {
    MPI_Status status;
    ierr =  MPI_Recv(&localval, 1, MPI_DOUBLE, rank-1, MPI_ANY_TAG,
             PETSC_COMM_WORLD, &status); CHKERRQ(ierr);
    localval /= rank;
    if (rank < size-1) {
      ierr = MPI_Send(&localval, 1, MPI_DOUBLE, rank+1, 1,
               PETSC_COMM_WORLD); CHKERRQ(ierr);
    }
  }

  // sum the contributions over all processes
  ierr = MPI_Allreduce(&localval, &globalsum, 1, MPI_DOUBLE, MPI_SUM,
                       PETSC_COMM_WORLD); CHKERRQ(ierr);

  // output one estimate of e
  ierr = PetscPrintf(PETSC_COMM_WORLD,
                     "e is about %17.15f\n",globalsum); CHKERRQ(ierr);

  // from each process, output report on work done
  ierr = PetscPrintf(PETSC_COMM_SELF,
                     "rank %d did %d flops\n",rank,rank == 0 ? 1 : 2); CHKERRQ(ierr);

  PetscFinalize();  // <-- always call
  return 0;
}
