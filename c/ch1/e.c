#include <petsc.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank;
  PetscInt       i;
  PetscReal      localval, globalsum;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute e in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  // compute  1/n!  where n = (rank of process) + 1
  localval = 1.0;
  for (i = 2; i < rank+1; i++)
      localval /= i;

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate of e and report on work from each process
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "e is about %17.15f\n",globalsum));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
      "rank %d did %d flops\n",rank,(rank > 0) ? rank-1 : 0));
  PetscCall(PetscFinalize());
  return 0;
}
