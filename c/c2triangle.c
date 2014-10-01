
static char help[] =
"Convert triangle-written ASCII mesh files to binary (demonstrates I/O with PETSc):.\n\
On the rank 0 process we do two steps:\n\
  1.  Read a FEM grid from ASCII files .node,.ele,.poly written by triangle.\n\
  2. Write in PETSc binary format (.petsc).\n\
Optionally, as a check, the binary file can be read in parallel and written to\n\
STDOUT.  In this case, node coordinates x,y have VecType \"mpi\" while integer\n\
index arrays BT,P,Q are sequential (and stored on all processes).\n\n\
For example, do:\n\
    triangle -pqa1.0 bump   # generate bump.1.{node,ele,poly}\n\
    c2triangle -f bump.1    # read bump.1.{node,ele,poly} and generate bump.1.petsc\n\n\
Do this to re-read binary file bump.1.petsc and show contents:\n\
    c2triangle -f bump.1 -check\n\n";

#include <petscmat.h>
#include "convenience.h"
#include "readmesh.h"

//STARTPREAMBLE
int main(int argc,char **args) {

  // STANDARD PREAMBLE
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD,
                  SELF = PETSC_COMM_SELF;
  PetscMPIInt     rank;
  MPI_Comm_rank(COMM,&rank);
  PetscErrorCode  ierr;
//ENDPREAMBLE

  // GET OPTIONS AND BUILD FILENAMES
  PetscBool      fset, docheck, checkset;
  const PetscInt  MPL = PETSC_MAX_PATH_LEN;
  char fnameroot[MPL], outfilename[MPL],
       nodefilename[MPL], elefilename[MPL], polyfilename[MPL];
  ierr = PetscOptionsBegin(COMM, "", "options for c2triangle", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root", "", "",
                            fnameroot, sizeof(fnameroot), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check", "if set, re-read and show at STDOUT", "", PETSC_FALSE,
                          &docheck,&checkset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(COMM,1,"option  -f FILENAMEROOT  required");
  }
  strcpy(nodefilename,fnameroot);  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);   strcat(elefilename,".ele");
  strcpy(polyfilename,fnameroot);  strcat(polyfilename,".poly");
  strcpy(outfilename,fnameroot);   strcat(outfilename,".petsc");
//ENDFILENAME

  // RANK 0 OPENS ASCII FILES
  FILE *nodefile, *elefile, *polyfile;
  ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
  ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);
  ierr = PetscFOpen(COMM,polyfilename,"r",&polyfile); CHKERRQ(ierr);

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
             "  N=%d nodes in 2D polygonal region\n"
             "  %d attributes and %d boundary markers per node\n",
             N,nattr,nbdrymarkers); CHKERRQ(ierr);

    // ALLOCATE 1D VECS FOR NODES
    Vec vx,vy,    // coordinates of N nodes
        vBT;      // boundary type of N nodes:
                  //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
    ierr = VecCreateSeq(SELF,N,&vx); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vy); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vBT); CHKERRQ(ierr);
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

    // READ ELEMENT HEADER
    PetscInt K,   // number of elements
             nthree, nattrele;
    if (3 != fscanf(elefile,"%d %d %d\n",&K,&nthree,&nattrele)) {
      SETERRQ1(SELF,2,"expected 3 values in reading from %s",elefilename);
    }
    if (nthree != 3) {
      SETERRQ1(SELF,3,"nthree read from %s not equal to 3",elefilename);
    }
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",elefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  K=%d elements in 2D polygonal region\n"
             "  %d attributes per element\n",
             K,nattrele); CHKERRQ(ierr);

    // ALLOCATE VEC FOR ELEMENTS
    Vec vP; // array with 3K rows; P[3*k+q] is node index (0 based)
    ierr = VecCreateSeq(SELF,3*K,&vP); CHKERRQ(ierr);

    // READ ELEMENTS; EACH PROCESS HOLDS FULL INFO (THREE NODE INDICES PER ELEMENT)
    PetscInt k, kplusone, q, j[3];
    PetscScalar w[3];
    for (k = 0; k < K; k++) {
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

    // READ POLYGON HEADER
    PetscInt M,   // number of segments in boundary polygon
             tmpa, tmpb, tmpc, tmpd;
    if (4 != fscanf(polyfile,"%d %d %d %d\n",&tmpa,&tmpb,&tmpc,&tmpd)) {
      SETERRQ1(SELF,2,"expected 4 values in reading from %s",polyfilename);
    }
    if (2 != fscanf(polyfile,"%d %d",&M,&tmpa)) {
      SETERRQ1(SELF,2,"expected 2 values in reading from %s",polyfilename);
    }
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",polyfilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,
             "  M=%d elements in boundary polygon\n",
             M); CHKERRQ(ierr);

    // ALLOCATE VEC FOR ELEMENTS
    Vec vQ; // array with 2M rows; Q[2*m+q] is node index (0 based)
    ierr = VecCreateSeq(SELF,2*M,&vQ); CHKERRQ(ierr);

    // READ POLYGON SEGMENT
    PetscInt m, mplusone;
    for (m = 0; m < M; m++) {
      if (4 != fscanf(polyfile,"%d %lf %lf %lf\n",&mplusone,&(w[0]),&(w[1]),&(w[2]))) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",polyfilename);
      }
      if (mplusone != m+1) {
        SETERRQ1(SELF,4,"indexing error in reading from %s",polyfilename);
      }
      for (q = 0; q < 2; q++)  {
        j[q] = 2*m + q;
        w[q]--;  // change to zero-based from triangle's default one-based
      }
      ierr = VecSetValues(vQ,2,j,w,INSERT_VALUES); CHKERRQ(ierr);
    }
    vecassembly(vQ)
//ENDREADPOLYGONS

    // DO BINARY WRITE
    PetscViewer viewer;
    ierr = PetscPrintf(SELF,"writing %s on rank 0 ...\n",outfilename); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,outfilename,FILE_MODE_WRITE,
               &viewer); CHKERRQ(ierr); 
    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    ierr = VecView(vBT,viewer); CHKERRQ(ierr);
    ierr = VecView(vP,viewer); CHKERRQ(ierr);
    ierr = VecView(vQ,viewer); CHKERRQ(ierr);

    // CLEAN UP
    VecDestroy(&vx);  VecDestroy(&vy);
    VecDestroy(&vBT);  VecDestroy(&vP);  VecDestroy(&vQ);
    PetscViewerDestroy(&viewer);
  }

  ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);
  ierr = PetscFClose(COMM,polyfile); CHKERRQ(ierr);
//ENDRANK0

  if (docheck == PETSC_TRUE) {
    ierr = PetscPrintf(COMM,"\nchecking %s by reading in Vecs and showing at STDOUT ...\n",
                       outfilename); CHKERRQ(ierr);

    // READ MESH FROM FILE
    PetscViewer rviewer;
    Vec rx,ry,rBT,rP,rQ;
    ierr = getmeshfile(COMM, outfilename, &rviewer); CHKERRQ(ierr);
    ierr = readmeshseqall(COMM, rviewer,
                          &rx, &ry, &rBT, &rP, &rQ); CHKERRQ(ierr);

    // SHOW WHAT WE GOT IF SMALL ENOUGH
    PetscInt N, bigsize=1000;
    ierr = VecGetSize(rx,&N); CHKERRQ(ierr);
    if (N < bigsize) {
      ierr = VecView(rx,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = VecView(ry,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = VecView(rBT,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = VecView(rP,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = VecView(rQ,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(COMM,"  [supressing STDOUT because too big]\n"); CHKERRQ(ierr);
    }

    // CLEAN UP
    VecDestroy(&rx);  VecDestroy(&ry);
    VecDestroy(&rBT);  VecDestroy(&rP);  VecDestroy(&rQ);
    PetscViewerDestroy(&rviewer);
  }

  PetscFinalize();
  return 0;
}

