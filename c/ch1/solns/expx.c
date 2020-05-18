#include <petsc.h>

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscReal      x = 1.0, localval, globalsum;

  PetscInitialize(&argc,&argv,NULL,
      "Compute exp(x) in parallel with PETSc.\n\n");
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  // read option
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","options for expx",""); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-x","input to exp(x) function",NULL,x,&x,NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // compute  x^n/n!  where n = (rank of process) + 1
  localval = 1.0;
  for (i = 1; i < rank+1; i++)
      localval *= x/i;

  // sum the contributions over all processes
  ierr = MPI_Allreduce(&localval,&globalsum,1,MPIU_REAL,MPIU_SUM,
      PETSC_COMM_WORLD); CHKERRQ(ierr);

  // output estimate and report on work from each process
  ierr = PetscPrintf(PETSC_COMM_WORLD,
      "exp(%17.15f) is about %17.15f\n",x,globalsum); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,
      "rank %d did %d flops\n",rank,(rank > 0) ? 2*rank : 0);
      CHKERRQ(ierr);
  return PetscFinalize();
}
