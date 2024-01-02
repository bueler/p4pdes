#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int rank, size;
  float localval, dx, x, globalsum;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  dx = 1.0 / size;
  x = rank * dx + dx / 2.0;
  localval = dx / (1.0 + x * x);

  MPI_Reduce(&localval, &globalsum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  /* // sum the contributions over all processes */
  /* MPI_Allreduce(&localval,&globalsum,1,MPI_REAL,MPI_SUM, MPI_COMM_WORLD); */
  if (rank == 0)
    printf("global sum: %17.15f\n", globalsum);
  // output estimate of e and report on work from each process
  MPI_Finalize();
  return 0;
}
