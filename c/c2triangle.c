static char help[] = "Read in a FEM grid from ASCII files written by triangle.\n\
Write out in PETSc binary format.\n\n";

// do
//    triangle -pqa1.0 bump
// (or similar) to generate bump.1.{node,ele}
// then do
//    c2triangle -f bump.1
// which reads bump.1.{node,ele} and writes bump.1.petsc

#include <petscmat.h>

#define DEBUG 1

int main(int argc,char **args)
{
  // MAJOR VARIABLES
  PetscInt N,   // number of degrees of freedom (= number of all nodes)
           M;   // number of elements;
  Vec vx,vy,    // coordinates of nodes
      vP,       // array with M rows and 3 columns; P[k][q] is node index
      vBT;      // array with N rows and 2 columns;
                //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann

  // INITIALIZE PETSC
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscMPIInt     rank;
  MPI_Comm_rank(COMM,&rank);
  PetscErrorCode  ierr;

  // GET FILENAME ROOT FROM OPTION
  const PetscInt MPL = PETSC_MAX_PATH_LEN;
  char           fnameroot[MPL];
  PetscBool      fset;
  ierr = PetscOptionsBegin(COMM, "", "options for c2triangle", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root", "", "",
                            fnameroot, sizeof(fnameroot), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAMEROOT  required");
  }

  // BUILD FILENAMES
  char           nodefilename[MPL], elefilename[MPL];
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");

  // RANK 0 OPENS ASCII FILES
  FILE           *nodefile, *elefile;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);

  if (rank == 0) {
    // READ NODE HEADER
    PetscInt ndim, nattr, nbdrymarkers;
    if (4 != fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers)) {
      SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
    }
    if (ndim != 2) {
      SETERRQ1(COMM,1,"ndim read from %s not equal to 2",nodefilename);
    }
    ierr = PetscPrintf(COMM,"read %s:\n",nodefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(COMM,
             "  N=%d nodes in 2D polygon with %d attributes and %d boundary markers per node\n",
             N,nattr,nbdrymarkers); CHKERRQ(ierr);

    // ALLOCATE 1D VECS
    ierr = VecCreateSeq(COMM,N,&vx); CHKERRQ(ierr);
    ierr = VecSetFromOptions(vx); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vy); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vBT); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vx,"node x coordinate"); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vy,"node y coordinate"); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vBT,"node boundary type"); CHKERRQ(ierr);

    // FIXME: deal with vP
    //ierr = PetscObjectSetName((PetscObject)vP,"element node index array"); CHKERRQ(ierr);

    // FILL 1D VECS FROM FILE BY READING NODE INFO
    double *ax, *ay, *aBT;
    ierr = VecGetArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(vBT,&aBT); CHKERRQ(ierr);
    PetscInt i,iplusone;
    for (i = 0; i < N; i++) {
      if (4 != fscanf(nodefile,"%d %lf %lf %lf\n",
                 &iplusone,&(ax[i]),&(ay[i]),&(aBT[i]))) {
        SETERRQ1(COMM,1,"expected 4 values in reading from %s",nodefilename);
      }
    }
#if DEBUG
    ierr = PetscPrintf(COMM,"boundary type read:\n"); CHKERRQ(ierr);
    for (i = 0; i < N; i++) {
      ierr = PetscPrintf(COMM,"%4d%6d\n",i,(int)aBT[i]); CHKERRQ(ierr);
    }
#endif
    ierr = VecRestoreArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(vBT,&aBT); CHKERRQ(ierr);

    // READ ELEMENT HEADER
    PetscInt nthree, nattrele;
    if (3 != fscanf(elefile,"%d %d %d\n",&M,&nthree,&nattrele)) {
      SETERRQ1(COMM,1,"expected 3 values in reading from %s",elefilename);
    }
    if (nthree != 3) {
      SETERRQ1(COMM,1,"nthree read from %s not equal to 3 (= nodes per element)",elefilename);
    }
    ierr = PetscPrintf(COMM,"read %s:\n",elefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(COMM,
             "  M=%d elements in 2D polygon with %d attributes per element\n",
             M,nattrele); CHKERRQ(ierr);

    // READ ELEMENTS; EACH PROCESS HOLDS FULL INFO (THREE NODE INDICES PER ELEMENT)
    PetscInt k, kplusone;
    int **P;
    PetscMalloc(M*sizeof(int*),&P);
    for (k = 0; k < M; k++) {
      PetscMalloc(3*sizeof(int),&(P[k]));
      if (4 != fscanf(elefile,"%d %d %d %d\n",&kplusone,&(P[k][0]),&(P[k][1]),&(P[k][2]))) {
        SETERRQ1(COMM,1,"expected 4 values in reading from %s",elefilename);
      }
    }
#if DEBUG
    ierr = PetscPrintf(COMM,"elements read:\n"); CHKERRQ(ierr);
    for (k = 0; k < M; k++) {
      ierr = PetscPrintf(COMM,"%4d%8d%6d%6d\n",k,P[k][0],P[k][1],P[k][2]); CHKERRQ(ierr);
    }
#endif

// FIXME: fill Vec vP 

    PetscFree(P);
  }
  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);

  // ALLOCATE VIEWER FOR BINARY WRITE, AND VECS
  PetscViewer viewer;
  char        outfilename[MPL];
  strcpy(outfilename,fnameroot);
  strcat(outfilename,".petsc");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfilename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);

  if (rank == 0) {
 
// FIXME: write Vecs in binary format
    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    ierr = VecView(vBT,viewer); CHKERRQ(ierr);

    VecDestroy(&vx);
    VecDestroy(&vy);
    VecDestroy(&vP);
    VecDestroy(&vBT);
  }

  PetscViewerDestroy(&viewer);
  PetscFinalize();
  return 0;
}
