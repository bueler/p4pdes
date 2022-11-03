#include <petsc.h>

int main(int argc, char **argv) {
  PetscMPIInt    rank;
  PetscInt       i;
  PetscReal      x = 1.0, localval, globalsum;

  PetscCall(PetscInitialize(&argc,&argv,NULL,
      "Compute exp(x) in parallel with PETSc.\n\n"));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  // read option
  PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx","");
  PetscCall(PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL));
  PetscOptionsEnd();

  // compute  x^n/n!  where n = (rank of process) + 1
  localval = 1.0;
  for (i = 1; i < rank+1; i++)
      localval *= x/i;

  // sum the contributions over all processes
  PetscCall(MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD));

  // output estimate and report on work from each process
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
      "exp(%17.15f) is about %17.15f\n",x,globalsum));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
      "rank %d did %d flops\n",rank,(rank > 0) ? 2*rank : 0));
  PetscCall(PetscFinalize());
  return 0;
}
