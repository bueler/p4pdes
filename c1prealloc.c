static char help[] = "Read in a FEM grid.  (Will evolve to showing Mat\n\
preallocation.)\n\n";

#include <petscmat.h>
#include <petscksp.h>

int main(int argc,char **args)
{
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm COMM = PETSC_COMM_WORLD;
  PetscErrorCode ierr;

  // do   triangle -pqa1.0 bump  (or similar) to generate bump.1.{node,ele}

  // FIXME use args for fnameroot
  FILE           *nodefile, *elefile;
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fnameroot[MPL], nodefilename[MPL], elefilename[MPL];
  strcpy(fnameroot,"bump.1");
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");

  PetscInt N, ndim, nattr, nbdrymarkers;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers); CHKERRQ(ierr);
  if (ndim != 2) {
    SETERRQ1(COMM,1,"ndim read from %s not equal to 2",nodefilename);
  }
  ierr = PetscPrintf(COMM,"read %s:\n",nodefilename); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "  N=%d nodes in 2D polygon with %d attributes and %d boundary markers per node\n",
           N,nattr,nbdrymarkers); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);

  PetscInt M, nthree, nattrele;
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);
  fscanf(nodefile,"%d %d %d\n",&M,&nthree,&nattrele); CHKERRQ(ierr);
  if (nthree != 3) {
    SETERRQ1(COMM,1,"nthree read from %s not equal to 3 (= nodes per element)",elefilename);
  }
  ierr = PetscPrintf(COMM,"read %s:\n",elefilename); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,
           "  M=%d elements in 2D polygon with %d attributes per element\n",
           M,nattrele); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);

// FIXME actually read the info so as to preallocate

  PetscFinalize();
  return 0;
}
