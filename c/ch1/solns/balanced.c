static char help[] = "Load-balanced computation of Euler's constant, with\n"
"a ridiculous amount of communication.\n\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscMPIInt     rank, size;
  PetscReal       localval, globalsum;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));  // <-- always call

  PetscCall(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  // compute  1 / n!  where n = (one more than rank of process)
  // by getting (n-1)! from previous processor
  if (rank == 0) {
    localval = 1.0;
    if (size > 1) {
      PetscCall(MPI_Send(&localval, 1, MPIU_REAL, rank+1, 1,
                         PETSC_COMM_WORLD));
    }
  } else {
    MPI_Status status;
    PetscCall(MPI_Recv(&localval, 1, MPIU_REAL, rank-1, MPI_ANY_TAG,
                       PETSC_COMM_WORLD, &status));
    localval /= rank;
    if (rank < size-1) {
      PetscCall(MPI_Send(&localval, 1, MPI_DOUBLE, rank+1, 1,
                         PETSC_COMM_WORLD));
    }
  }

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval, &globalsum, 1, MPIU_REAL, MPIU_SUM,
                          PETSC_COMM_WORLD));

  // output one estimate of e
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "e is about %17.15f\n",globalsum));

  // from each process, output report on work done
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
                        "rank %d did %d flops\n",rank,rank == 0 ? 1 : 2));

  PetscCall(PetscFinalize());
  return 0;
}

