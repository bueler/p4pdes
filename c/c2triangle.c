static char help[] = 
   "Read (rank 0 only) a FEM grid from ASCII files written by triangle.\n\
    Write (rank 0 only) in PETSc binary format.  Read (all ranks) back\n\
    and show at stdout.\n\n";

// do
//    triangle -pqa1.0 bump
// (or similar) to generate bump.1.{node,ele}
// then do
//    c2triangle -f bump.1
// which reads bump.1.{node,ele} and writes bump.1.petsc

#include <petscmat.h>

#define DEBUG 0

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD,
                  SELF = PETSC_COMM_SELF;
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
  char outfilename[MPL], nodefilename[MPL], elefilename[MPL];
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");
  strcpy(outfilename,fnameroot);
  strcat(outfilename,".petsc");

  // RANK 0 OPENS ASCII FILES
  FILE *nodefile, *elefile;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);

  if (rank == 0) {
    // READ NODE HEADER
    PetscInt N,   // number of degrees of freedom (= number of all nodes)
             ndim, nattr, nbdrymarkers;
    if (4 != fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers)) {
      SETERRQ1(SELF,1,"expected 4 values in reading from %s",nodefilename);
    }
    if (ndim != 2) {
      SETERRQ1(SELF,1,"ndim read from %s not equal to 2",nodefilename);
    }
    ierr = PetscPrintf(SELF,"read %s on rank 0:\n",nodefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  N=%d nodes in 2D polygon\n"
             "  %d attributes and %d boundary markers per node\n",
             N,nattr,nbdrymarkers); CHKERRQ(ierr);

    // ALLOCATE 1D VECS
    Vec vx,vy,    // coordinates of nodes
        vP,       // array with M rows and 3 columns; P[k][q] is node index
        vBT;      // array with N rows and 2 columns;
                  //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
    ierr = VecCreateSeq(SELF,N,&vx); CHKERRQ(ierr);
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
        SETERRQ1(SELF,1,"expected 4 values in reading from %s",nodefilename);
      }
    }
#if DEBUG
    ierr = PetscPrintf(SELF,"node location and boundary type read:\n"); CHKERRQ(ierr);
    for (i = 0; i < N; i++) {
      ierr = PetscPrintf(SELF,"%4d%14.8f%14.8f%6d\n",i,ax[i],ay[i],(int)aBT[i]); CHKERRQ(ierr);
    }
#endif
    ierr = VecRestoreArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(vBT,&aBT); CHKERRQ(ierr);

    // READ ELEMENT HEADER
    PetscInt M,   // number of elements
             nthree, nattrele;
    if (3 != fscanf(elefile,"%d %d %d\n",&M,&nthree,&nattrele)) {
      SETERRQ1(SELF,1,"expected 3 values in reading from %s",elefilename);
    }
    if (nthree != 3) {
      SETERRQ1(SELF,1,"nthree read from %s not equal to 3 (= nodes per element)",elefilename);
    }
    ierr = PetscPrintf(SELF,"read %s on rank 0:\n",elefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  M=%d elements in 2D polygon\n"
             "  %d attributes per element\n",
             M,nattrele); CHKERRQ(ierr);

    // READ ELEMENTS; EACH PROCESS HOLDS FULL INFO (THREE NODE INDICES PER ELEMENT)
    PetscInt k, kplusone;
    int **P;
    PetscMalloc(M*sizeof(int*),&P);
    for (k = 0; k < M; k++) {
      PetscMalloc(3*sizeof(int),&(P[k]));
      if (4 != fscanf(elefile,"%d %d %d %d\n",&kplusone,&(P[k][0]),&(P[k][1]),&(P[k][2]))) {
        SETERRQ1(SELF,1,"expected 4 values in reading from %s",elefilename);
      }
    }
#if DEBUG
    ierr = PetscPrintf(SELF,"elements read:\n"); CHKERRQ(ierr);
    for (k = 0; k < M; k++) {
      ierr = PetscPrintf(SELF,"%4d%8d%6d%6d\n",k,P[k][0],P[k][1],P[k][2]); CHKERRQ(ierr);
    }
#endif

    // ALLOCATE VIEWER FOR BINARY WRITE, AND VECS
    PetscViewer viewer;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,outfilename,FILE_MODE_WRITE,
               &viewer); CHKERRQ(ierr); 

    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    ierr = VecView(vBT,viewer); CHKERRQ(ierr);
#if DEBUG
    ierr = VecView(vx,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
#endif

    VecDestroy(&vx);
    VecDestroy(&vy);
    VecDestroy(&vP);
    VecDestroy(&vBT);

    PetscViewerDestroy(&viewer);

// FIXME: fill Vec vP 

    PetscFree(P);
  }

  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);

  // READ BACK IN PARALLEL: ALLOCATE VIEWER AND VECS
  Vec rx, ry, rBT;
  //Vec rx,ry,rP,rBT;
  PetscViewer rviewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfilename,FILE_MODE_READ,
             &rviewer); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&rx); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&ry); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&rBT); CHKERRQ(ierr);

  // FILL 1D VECS FROM FILE
  ierr = PetscPrintf(COMM,"reading from %s in parallel:\n",outfilename); CHKERRQ(ierr);
  ierr = VecLoad(rx,rviewer); CHKERRQ(ierr);
  ierr = VecLoad(ry,rviewer); CHKERRQ(ierr);
  ierr = VecLoad(rBT,rviewer); CHKERRQ(ierr);

  PetscInt rN;
  ierr = VecGetSize(rx,&rN); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  N=%d nodes\n",rN); CHKERRQ(ierr);

  ierr = VecView(rx,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecView(ry,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = VecView(rBT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

  VecDestroy(&rx);
  VecDestroy(&ry);
  VecDestroy(&rBT);
  //VecDestroy(&rP);
  PetscViewerDestroy(&rviewer);

  PetscFinalize();
  return 0;
}
