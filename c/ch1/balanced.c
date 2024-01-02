#include <petsc.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank, size;
  PetscReal      localval, globalsum;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute e in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  // compute  1/n!  where n = (rank of process) + 1
  if (rank == 0)
    localval = 1.0;
  else
    {
      PetscCall(MPI_Recv(&localval, 1, MPIU_REAL, rank - 1, 0,  PETSC_COMM_WORLD, MPI_STATUS_IGNORE));
      localval /= rank;
    }
  if (rank != (size - 1))
    PetscCall(MPI_Send(&localval, 1, MPIU_REAL, rank + 1, 0, PETSC_COMM_WORLD));

    // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate of e and report on work from each process
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "e is about %17.15f\n",globalsum));
  /* PetscCall(PetscPrintf(PETSC_COMM_SELF, */
  /*     "rank %d did %d flops\n",rank,(rank > 0) ? rank-1 : 0)); */
  PetscCall(PetscFinalize());
  return 0;
}
