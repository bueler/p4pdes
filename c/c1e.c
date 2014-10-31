// regarding this idea, see
//   http://lists.mcs.anl.gov/pipermail/petsc-dev/2010-May/002812.html

static char help[] = "To get started with PETSc: compute e in parallel.\n\n";

#include <petsc.h>

int main(int argc,char **args) {
  PetscErrorCode  ierr;

  PetscInitialize(&argc,&args,(char*)0,help);
  PetscMPIInt     rank, size;
  PetscScalar     localsum, globalsum;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);

  localsum = (double)rank;  // FIXME

  //ierr = PetscGlobalSum(PETSC_COMM_WORLD,&localsum,&globalsum); CHKERRQ(ierr);
  MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"sum of ranks is %f\n",globalsum); CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
//END
