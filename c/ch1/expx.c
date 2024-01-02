#include <petsc.h>
#include <math.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank;
  PetscInt       i;
  PetscReal      localval, globalsum;
  PetscReal      x;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute the Maclaurin series of x\n\n"));

  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx","");
  PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL);
  PetscOptionsEnd();

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "x is %17.15f\n", x));

  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  // compute  1/n!  where n = (rank of process) + 1
  localval = 1.0;
  for (i = rank; i > 1; i--)
    localval *= i;
  localval = pow(x, rank) / localval;

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate of e and report on work from each process
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "sum is about %17.15f\n",globalsum));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
                        "localval on %d: %17.15f\n",rank, localval));
  PetscCall(PetscFinalize());
  return 0;
}
