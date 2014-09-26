
static char help[] =
"Read (rank 0 only) a FEM grid from ASCII files written by triangle.\n\
Write (rank 0 only) in PETSc binary format.\n\
Read (all ranks) back and show at stdout.\n\n\
For example, do\n\
    triangle -pqa1.0 bump\n\
(or similar) to generate bump.1.{node,ele}.\n\
Then do\n\
    c2triangle -f bump.1\n\
which reads bump.1.{node,ele} and writes bump.1.petsc.\n\n";

#include <petscmat.h>
#define DEBUG 0
#define vecassembly(X) { ierr = VecAssemblyBegin(X); CHKERRQ(ierr); \
                         ierr = VecAssemblyEnd(X); CHKERRQ(ierr); }

int main(int argc,char **args) {

  // STANDARD PREAMBLE
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD,
                  SELF = PETSC_COMM_SELF;
  PetscMPIInt     rank;
  MPI_Comm_rank(COMM,&rank);
  const PetscInt  MPL = PETSC_MAX_PATH_LEN;
  PetscErrorCode  ierr;
//ENDPREAMBLE

  // GET FILENAME ROOT FROM OPTION AND BUILD FILENAMES
  PetscBool      fset;
  char fnameroot[MPL], outfilename[MPL], nodefilename[MPL], elefilename[MPL];
  ierr = PetscOptionsBegin(COMM, "", "options for c2triangle", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root", "", "",
                            fnameroot, sizeof(fnameroot), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAMEROOT  required");
  }
  strcpy(nodefilename,fnameroot);
  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);
  strcat(elefilename,".ele");
  strcpy(outfilename,fnameroot);
  strcat(outfilename,".petsc");
//ENDFILENAME

  // RANK 0 OPENS ASCII FILES
  FILE *nodefile, *elefile;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);

  if (rank == 0) {
    // READ NODE HEADER
    PetscInt N,   // number of degrees of freedom (= number of all nodes)
             ndim, nattr, nbdrymarkers;
    if (4 != fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers)) {
      SETERRQ1(SELF,2,"expected 4 values in reading from %s",nodefilename);
    }
    if (ndim != 2) {
      SETERRQ1(SELF,3,"ndim read from %s not equal to 2",nodefilename);
    }
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",nodefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  N=%d nodes in 2D polygon\n"
             "  %d attributes and %d boundary markers per node\n",
             N,nattr,nbdrymarkers); CHKERRQ(ierr);

    // ALLOCATE 1D VECS FOR NODES
    Vec vx,vy,    // coordinates of N nodes
        vBT;      // boundary type of N nodes:
                  //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
    ierr = VecCreateSeq(SELF,N,&vx); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vy); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vBT); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vx,"node x coordinate (seq)"); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vy,"node y coordinate (seq)"); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vBT,"node boundary type (seq)"); CHKERRQ(ierr);
//ENDRANK0ALLOC

    // FILL 1D VECS FROM NODE FILE
    PetscInt    i,iplusone;
    PetscScalar v[3];
    for (i = 0; i < N; i++) {
      if (4 != fscanf(nodefile,"%d %lf %lf %lf\n",
                 &iplusone,&(v[0]),&(v[1]),&(v[2]))) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",nodefilename);
      }
      if (iplusone != i+1) {
        SETERRQ1(SELF,4,"indexing error in reading from %s",nodefilename);
      }
      ierr = VecSetValues(vx,1,&i,&(v[0]),INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValues(vy,1,&i,&(v[1]),INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValues(vBT,1,&i,&(v[2]),INSERT_VALUES); CHKERRQ(ierr);
    }
    vecassembly(vx)
    vecassembly(vy)
    vecassembly(vBT)
//ENDREADNODES
#if DEBUG
    ierr = PetscPrintf(SELF,"node location and boundary type read (DEBUG):\n"); CHKERRQ(ierr);
    ierr = VecView(vx,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    ierr = VecView(vy,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
    ierr = VecView(vBT,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
#endif

//STARTREADELEMENTS
    // READ ELEMENT HEADER
    PetscInt M,   // number of elements
             nthree, nattrele;
    if (3 != fscanf(elefile,"%d %d %d\n",&M,&nthree,&nattrele)) {
      SETERRQ1(SELF,2,"expected 3 values in reading from %s",elefilename);
    }
    if (nthree != 3) {
      SETERRQ1(SELF,3,"nthree read from %s not equal to 3 (= nodes per element)",elefilename);
    }
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",elefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  M=%d elements in 2D polygon\n"
             "  %d attributes per element\n",
             M,nattrele); CHKERRQ(ierr);

    // ALLOCATE VEC FOR ELEMENTS
    Vec vP; // array with 3M rows; P[k][q] is node index (0 based)
    ierr = VecCreateSeq(SELF,3*M,&vP); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)vP,"element node index array (seq and 0 based)"); CHKERRQ(ierr);

    // READ ELEMENTS; EACH PROCESS HOLDS FULL INFO (THREE NODE INDICES PER ELEMENT)
    PetscInt k, kplusone, q, j[3];
    PetscScalar w[3];
    for (k = 0; k < M; k++) {
      if (4 != fscanf(elefile,"%d %lf %lf %lf\n",&kplusone,&(w[0]),&(w[1]),&(w[2]))) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",elefilename);
      }
      if (kplusone != k+1) {
        SETERRQ1(SELF,4,"indexing error in reading from %s",elefilename);
      }
      for (q = 0; q < 3; q++)  {
        j[q] = 3*k + q;
        w[q]--;  // change to zero-based from triangle's default one-based
      }
      ierr = VecSetValues(vP,3,j,w,INSERT_VALUES); CHKERRQ(ierr);
    }
    vecassembly(vP)
//ENDREADELEMENTS
#if DEBUG
    ierr = PetscPrintf(SELF,"element indices read (DEBUG):\n"); CHKERRQ(ierr);
    ierr = VecView(vP,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
#endif

//STARTBINARYWRITE
    // DO BINARY WRITE
    PetscViewer viewer;
    ierr = PetscPrintf(SELF,"writing %s on rank 0 ...\n",outfilename); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,outfilename,FILE_MODE_WRITE,
               &viewer); CHKERRQ(ierr); 
    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    ierr = VecView(vBT,viewer); CHKERRQ(ierr);
    ierr = VecView(vP,viewer); CHKERRQ(ierr);

    // CLEAN UP
    VecDestroy(&vx);
    VecDestroy(&vy);
    VecDestroy(&vBT);
    VecDestroy(&vP);
    PetscViewerDestroy(&viewer);
  }

  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);
//ENDRANK0

  // READ BACK IN PARALLEL: ALLOCATE VIEWER AND VECS
  Vec rx,ry,rP,rBT;
  PetscViewer rviewer;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outfilename,FILE_MODE_READ,
             &rviewer); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&rx); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&ry); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&rBT); CHKERRQ(ierr);
  ierr = VecCreate(COMM,&rP); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rx,"node x coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)ry,"node y coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rBT,"node boundary type"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)rP,"element node index array (0 based)"); CHKERRQ(ierr);

  // FILL FROM FILE
  ierr = PetscPrintf(COMM,"reading from %s in parallel ...\n",outfilename); CHKERRQ(ierr);
  ierr = VecLoad(rx,rviewer); CHKERRQ(ierr);
  ierr = VecLoad(ry,rviewer); CHKERRQ(ierr);
  ierr = VecLoad(rBT,rviewer); CHKERRQ(ierr);
  ierr = VecLoad(rP,rviewer); CHKERRQ(ierr);

  // SHOW WHAT WE GOT
  PetscInt rN, rM, bigsize=1000;
  ierr = VecGetSize(rx,&rN); CHKERRQ(ierr);
  ierr = VecGetSize(rP,&rM); CHKERRQ(ierr);
  rM /= 3;
  ierr = PetscPrintf(COMM,"  N=%d nodes, M=%d elements\n",rN,rM); CHKERRQ(ierr);
  if ((rN < bigsize) && (rM < bigsize)) {
    ierr = VecView(rx,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(ry,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(rBT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(rP,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(COMM,"  [supressing VecView to STDOUT because too big]\n"); CHKERRQ(ierr);
  }

  // CLEAN UP
  VecDestroy(&rx);
  VecDestroy(&ry);
  VecDestroy(&rBT);
  VecDestroy(&rP);
  PetscViewerDestroy(&rviewer);

  PetscFinalize();
  return 0;
}
//END
