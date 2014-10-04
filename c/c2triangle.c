
static char help[] =
"Convert triangle-written ASCII mesh files to binary PETSc files.\n\
On the rank 0 process we do two steps:\n\
  1. Read a FEM grid from ASCII files .node,.ele,.poly written by triangle.\n\
  2. Write elements in PETSc binary format (.Epetsc).\n\
  3. Write nodal info in PETSc binary format (.Npetsc).\n\
Optionally, as a check, the binary file can be read in parallel and written to\n\
STDOUT.  For example, do:\n\
    triangle -pqa1.0 bump   # generate bump.1.{node,ele,poly}\n\
    c2triangle -f bump.1    # read bump.1.{node,ele,poly}\n\
                            # and generate bump.1.{Epetsc,.Npetsc}\n\n\
Do this to re-read binary files bump.1.{Epetsc,Npetsc} and show contents:\n\
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
  char fnameroot[MPL], Eoutfilename[MPL], Noutfilename[MPL],
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
  strcpy(Eoutfilename,fnameroot);   strcat(Eoutfilename,".Epetsc");
  strcpy(Noutfilename,fnameroot);   strcat(Noutfilename,".Npetsc");
//ENDFILENAME

  if (rank == 0) {
    FILE *nodefile, *elefile, *polyfile;

    // READ NODE HEADER AND ALLOCATE VECS
    Vec vx,vy,    // coordinates of N nodes
        vBT;      // boundary type of N nodes:
                  //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
    ierr = PetscFOpen(COMM,nodefilename,"r",&nodefile); CHKERRQ(ierr);
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
    ierr = VecCreateSeq(SELF,N,&vx); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vy); CHKERRQ(ierr);
    ierr = VecDuplicate(vx,&vBT); CHKERRQ(ierr);

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
    ierr = PetscFClose(COMM,nodefile); CHKERRQ(ierr);
    vecassembly(vx)
    vecassembly(vy)
    vecassembly(vBT)
//ENDREADNODES

    // READ POLYGON HEADER AND ALLOCATE VEC
    Vec vQ; // array with 2M rows; Q[2*m+q] is node index (0 based)
    ierr = PetscFOpen(COMM,polyfilename,"r",&polyfile); CHKERRQ(ierr);
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
    ierr = VecCreateSeq(SELF,2*M,&vQ); CHKERRQ(ierr);
    ierr = VecSetBlockSize(vQ,2); CHKERRQ(ierr);

    // READ POLYGON SEGMENT
    PetscInt m, mplusone, q;
    PetscScalar w[2], discard;
    for (m = 0; m < M; m++) {
      if (4 != fscanf(polyfile,"%d %lf %lf %lf\n",&mplusone,&(w[0]),&(w[1]),&discard)) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",polyfilename);
      }
      if (mplusone != m+1) {
        SETERRQ1(SELF,4,"indexing error in reading from %s",polyfilename);
      }
      for (q = 0; q < 2; q++)  {
        w[q]--;  // change to zero-based from triangle's default one-based
      }
      ierr = VecSetValuesBlocked(vQ,1,&m,w,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = PetscFClose(COMM,polyfile); CHKERRQ(ierr);
    vecassembly(vQ)
//ENDREADPOLYGONS

    // READ ELEMENT HEADER AND ALLOCATE VEC
    //   vE[k] is a 12 scalar struct with full info on element k:
    typedef struct {
      PetscScalar j[3],  // global indices of nodes j[0], j[1], j[2]
                  BT[3], // boundary type of node:  BT[0], BT[1], BT[2]
                  x[3],  // node x-coordinate x[0], x[1], x[2]
                  y[3];  // node y-coordinate y[0], y[1], y[2]
    } element;
    Vec vE;
    ierr = PetscFOpen(COMM,elefilename,"r",&elefile); CHKERRQ(ierr);
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
    ierr = VecCreateSeq(SELF,12*K,&vE); CHKERRQ(ierr);
    ierr = VecSetBlockSize(vE,12); CHKERRQ(ierr);

    // READ ELEMENTS
    element e;
    PetscInt k, kplusone;
    PetscScalar *ax, *ay, *aBT;
    ierr = VecGetArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(vBT,&aBT); CHKERRQ(ierr);
    for (k = 0; k < K; k++) {
      if (4 != fscanf(elefile, "%d %lf %lf %lf\n",
                      &kplusone, &(e.j[0]), &(e.j[1]), &(e.j[2]))) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",elefilename);
      }
      if (kplusone != k+1) {
        SETERRQ1(SELF,4,"indexing error in reading from %s",elefilename);
      }
      for (q = 0; q < 3; q++)  {
        e.j[q]--; // change to zero-based from triangle's default one-based
        e.BT[q] = aBT[(int)e.j[q]];
        e.x[q] = ax[(int)e.j[q]];
        e.y[q] = ay[(int)e.j[q]];
      }
      ierr = VecSetValuesBlocked(vE,1,&k,(PetscScalar*)&e,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(vBT,&aBT); CHKERRQ(ierr);
    ierr = PetscFClose(COMM,elefile); CHKERRQ(ierr);
    VecDestroy(&vBT);

    vecassembly(vE)
//ENDREADELEMENTS

    // DO BINARY WRITE AND CLEAN UP
    PetscViewer viewer;
    ierr = PetscPrintf(SELF,"writing %s on rank 0 ...\n",Eoutfilename); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,Eoutfilename,FILE_MODE_WRITE,
               &viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vE,"E");
    ierr = VecView(vE,viewer); CHKERRQ(ierr);
    VecDestroy(&vE);
    PetscViewerDestroy(&viewer);
    ierr = PetscPrintf(SELF,"writing %s on rank 0 ...\n",Noutfilename); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,Noutfilename,FILE_MODE_WRITE,
               &viewer); CHKERRQ(ierr); 
    VecSetOptionsPrefix(vx,"x");
    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vy,"y");
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vQ,"Q");
    ierr = VecView(vQ,viewer); CHKERRQ(ierr);
    VecDestroy(&vx);  VecDestroy(&vy);  VecDestroy(&vQ);
    PetscViewerDestroy(&viewer);
  }
//ENDRANK0

  if (docheck == PETSC_TRUE) {
    ierr = PetscPrintf(COMM,"\nchecking by reading in Vecs and showing at STDOUT ...\n"
                       ); CHKERRQ(ierr);

    // READ MESH FROM FILE, SHOW IT, AND CLEAN UP
    PetscViewer Eviewer,Nviewer;
    Vec rE,rx,ry,rQ;
    ierr = getmeshfile(COMM, ".Epetsc", Eoutfilename, &Eviewer); CHKERRQ(ierr);
    ierr = getmeshfile(COMM, ".Npetsc", Noutfilename, &Nviewer); CHKERRQ(ierr);
    ierr = readmesh(COMM, Eviewer, Nviewer,
                    &rE, &rx, &ry, &rQ); CHKERRQ(ierr);
    ierr = VecView(rE,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(rx,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(ry,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(rQ,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    VecDestroy(&rE);  VecDestroy(&rx);  VecDestroy(&ry);  VecDestroy(&rQ);
    PetscViewerDestroy(&Eviewer);   PetscViewerDestroy(&Nviewer);
  }

  PetscFinalize();
  return 0;
}

