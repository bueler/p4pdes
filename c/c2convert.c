
static char help[] =
"Convert triangle-written ASCII mesh files to a binary PETSc file.\n\
On the rank 0 process we do two steps:\n\
  1. Read a triangular mesh from ASCII files .node,.ele,.poly (from triangle).\n\
  2. Write elements and nodal info in PETSc binary format (.petsc).\n\
Optionally, as a check, the binary file can be read in parallel and written to\n\
STDOUT.  For example, do:\n\
    triangle -pqa1.0 bump  # generate bump.1.{node,ele,poly}\n\
    c2convert -f bump.1    # read bump.1.{node,ele,poly}; generate bump.1.petsc\n\n\
Do this to re-read binary files bump.1.petsc and show contents:\n\
    c2convert -f bump.1 -check\n\n";

#include <petscmat.h>
#include "convenience.h"
#include "readmesh.h"

//STARTPREAMBLE
int main(int argc,char **args) {
  PetscErrorCode  ierr;  //STRIP

  // STANDARD PREAMBLE
  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  WORLD = PETSC_COMM_WORLD,  SELF = PETSC_COMM_SELF;
  PetscMPIInt     rank;
  MPI_Comm_rank(WORLD,&rank);

  // GET OPTIONS AND BUILD FILENAMES
  PetscBool      fset, docheck, checkset;
  const PetscInt  MPL = PETSC_MAX_PATH_LEN;
  char fnameroot[MPL], outfilename[MPL],
       nodefilename[MPL], elefilename[MPL], polyfilename[MPL];
  ierr = PetscOptionsBegin(WORLD, "", "options for c2convert", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root", "", "",
                            fnameroot, sizeof(fnameroot), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-check", "if set, re-read and show at STDOUT", "", PETSC_FALSE,
                          &docheck,&checkset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {  SETERRQ(WORLD,1,"option  -f FILENAMEROOT  required");  } //STRIP
  strcpy(nodefilename,fnameroot);  strcat(nodefilename,".node");
  strcpy(elefilename,fnameroot);   strcat(elefilename,".ele");
  strcpy(polyfilename,fnameroot);  strcat(polyfilename,".poly");
  strcpy(outfilename,fnameroot);   strcat(outfilename,".petsc");
//ENDFILENAME

  if (rank == 0) {
    FILE *nodefile, *elefile, *polyfile;

    // READ NODE HEADER AND ALLOCATE VECS
    Vec vx,vy,    // coordinates of N nodes
        vBT;      // boundary type of N nodes:
                  //   BT[i] = 0 if interior, 2 if Dirichlet, 3 if Neumann
    ierr = PetscFOpen(WORLD,nodefilename,"r",&nodefile); CHKERRQ(ierr);
    PetscInt N,   // number of degrees of freedom (= number of all nodes)
             ndim, nattr, nbdrymarkers;
    if (4 != fscanf(nodefile,"%d %d %d %d\n",&N,&ndim,&nattr,&nbdrymarkers)) {
      SETERRQ1(SELF,2,"expected 4 values in reading from %s",nodefilename);  }
    if (ndim != 2) {  //STRIP
      SETERRQ1(SELF,3,"ndim read from %s not equal to 2",nodefilename);  }  //STRIP
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",nodefilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,"  N=%d nodes in 2D polygonal region\n"
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
          SETERRQ1(SELF,2,"expected 4 values in reading from %s",nodefilename);  }
      if (iplusone != i+1) {   //STRIP
          SETERRQ1(SELF,4,"indexing error in reading from %s",nodefilename);  }  //STRIP
      ierr = VecSetValues(vx,1,&i,&(v[0]),INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValues(vy,1,&i,&(v[1]),INSERT_VALUES); CHKERRQ(ierr);
      ierr = VecSetValues(vBT,1,&i,&(v[2]),INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = PetscFClose(WORLD,nodefile); CHKERRQ(ierr);
    vecassembly(vx)
    vecassembly(vy)
    vecassembly(vBT)
//ENDREADNODES

    // READ POLYGON HEADER AND ALLOCATE VEC
    Vec vQ; // array with 2M rows; Q[2*m+q] is node index (0 based)
    ierr = PetscFOpen(WORLD,polyfilename,"r",&polyfile); CHKERRQ(ierr);
    PetscInt M,   // number of segments in boundary polygon
             tmpa, tmpb, tmpc, tmpd;
    if (4 != fscanf(polyfile,"%d %d %d %d\n",&tmpa,&tmpb,&tmpc,&tmpd)) {
        SETERRQ1(SELF,2,"expected 4 values in reading from %s",polyfilename);  }
    if (2 != fscanf(polyfile,"%d %d",&M,&tmpa)) {
        SETERRQ1(SELF,2,"expected 2 values in reading from %s",polyfilename);  }
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n",polyfilename); CHKERRQ(ierr);
    ierr = PetscPrintf(SELF,"  M=%d elements in boundary polygon\n",M); CHKERRQ(ierr);
    ierr = VecCreateSeq(SELF,2*M,&vQ); CHKERRQ(ierr);
    ierr = VecSetBlockSize(vQ,2); CHKERRQ(ierr);

    // READ POLYGON SEGMENT
    PetscInt m, mplusone, q;
    PetscScalar w[2], discard;
    for (m = 0; m < M; m++) {
      if (4 != fscanf(polyfile,"%d %lf %lf %lf\n",&mplusone,&(w[0]),&(w[1]),&discard)) {
          SETERRQ1(SELF,2,"expected 4 values in reading from %s",polyfilename);  }
      if (mplusone != m+1) {  //STRIP
          SETERRQ1(SELF,4,"indexing error in reading from %s",polyfilename);  }  //STRIP
      for (q = 0; q < 2; q++)  w[q]--;  // change to zero-based
      ierr = VecSetValuesBlocked(vQ,1,&m,w,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = PetscFClose(WORLD,polyfile); CHKERRQ(ierr);
    vecassembly(vQ)
//ENDREADPOLYGONS

    // READ ELEMENT HEADER AND ALLOCATE VEC
    // vE[k] is an elementtype struct, with full info on element k
    Vec vE;
    ierr = PetscFOpen(WORLD,elefilename,"r",&elefile); CHKERRQ(ierr);
    PetscInt K, nthree, nattrele;  // K = number of elements
    if (3 != fscanf(elefile,"%d %d %d\n",&K,&nthree,&nattrele)) {
        SETERRQ1(SELF,2,"expected 3 values in reading from %s",elefilename);  }
    if (nthree != 3) {  //STRIP
        SETERRQ1(SELF,3,"nthree read from %s not equal to 3",elefilename);  }  //STRIP
    ierr = PetscPrintf(SELF,"reading %s on rank 0 ...\n"
                     "  K=%d elements in 2D polygonal region\n"
                     "  %d attributes per element\n",
                     elefilename,K,nattrele); CHKERRQ(ierr);
    ierr = VecCreateSeq(SELF,15*K,&vE); CHKERRQ(ierr);
    ierr = VecSetBlockSize(vE,15); CHKERRQ(ierr);

//STARTREADELEMENTS
    // READ ELEMENTS AND CREATE VEC vE
    elementtype e;
    PetscInt    k, kplusone, l, qnext;
    PetscScalar *ax, *ay, *aBT, *aQ;
    ierr = VecGetArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecGetArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecGetArray(vBT,&aBT); CHKERRQ(ierr);
    ierr = VecGetArray(vQ,&aQ); CHKERRQ(ierr);
    for (k = 0; k < K; k++) {
      if (4 != fscanf(elefile, "%d %lf %lf %lf\n",
                      &kplusone, &(e.j[0]), &(e.j[1]), &(e.j[2]))) {
          SETERRQ1(SELF,2,"expected 4 values in reading from %s",elefilename);  }
      if (kplusone != k+1) {  //STRIP
        SETERRQ1(SELF,4,"indexing error in reading from %s",elefilename);  }  //STRIP
      for (q = 0; q < 3; q++)  {
        e.j[q]--;                   // change to zero-based
        e.bN[q] = aBT[(int)e.j[q]]; // node boundary type; in {0,2,3}
        e.bE[q] = 0;                // will compute it below; this sets it to not-boundary
        e.x[q] = ax[(int)e.j[q]];
        e.y[q] = ay[(int)e.j[q]];
      }
      if (e.bN[0] + e.bN[1] + e.bN[2] > 3.5) { // element has >= two nodes on boundary
        for (q = 0; q < 3; q++)  {
          qnext = (q < 2) ? q+1 : 0;           // cycle
          if ((e.bN[q] > 0) && (e.bN[qnext] > 0)) {
            // end-nodes of this edge are on boundary; is it an edge
            //   in the polygon (= boundary segment list)?
            const PetscInt ja = (int)e.j[q],
                           jb = (int)e.j[qnext];
            for (l = 0; l < M; l++) {
              const PetscInt qa = (int)(aQ[2*l + 0]),
                             qb = (int)(aQ[2*l + 1]);
              if (  ((ja == qa) && (jb == qb)) || ((ja == qb) && (jb == qa))  ) {
                e.bE[q] = 1.0;
                break;
              }
            }
          }
        }
      }
      ierr = VecSetValuesBlocked(vE,1,&k,(PetscScalar*)&e,INSERT_VALUES); CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(vx,&ax); CHKERRQ(ierr);
    ierr = VecRestoreArray(vy,&ay); CHKERRQ(ierr);
    ierr = VecRestoreArray(vBT,&aBT); CHKERRQ(ierr);
    ierr = VecRestoreArray(vQ,&aQ); CHKERRQ(ierr);
    ierr = PetscFClose(WORLD,elefile); CHKERRQ(ierr);
    vecassembly(vE)
//ENDREADELEMENTS

    // DO BINARY WRITE AND CLEAN UP
    PetscViewer viewer;
    ierr = PetscPrintf(SELF,"writing %s on rank 0 ...\n",outfilename); CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(SELF,outfilename,FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vE,"E_");
    ierr = VecView(vE,viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vx,"x_");
    ierr = VecView(vx,viewer); CHKERRQ(ierr);
    VecSetOptionsPrefix(vy,"y_");
    ierr = VecView(vy,viewer); CHKERRQ(ierr);
    VecDestroy(&vBT);  VecDestroy(&vQ);
    VecDestroy(&vE);  VecDestroy(&vx);  VecDestroy(&vy);
    PetscViewerDestroy(&viewer);
  } // end if (rank == 0)
//ENDRANK0

  if (docheck == PETSC_TRUE) {
    PetscViewer viewer;
    Vec rE,rx,ry;
    ierr = PetscPrintf(WORLD,"\nchecking by loading Vecs and viewing at STDOUT ...\n"
                       ); CHKERRQ(ierr);
    ierr = getmeshfile(WORLD, ".petsc", outfilename, &viewer); CHKERRQ(ierr);
    ierr = readmesh(WORLD, viewer, &rE, &rx, &ry); CHKERRQ(ierr);
    ierr = elementVecViewSTDOUT(WORLD, rE); CHKERRQ(ierr);
    ierr = VecView(rx,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = VecView(ry,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    VecDestroy(&rE);  VecDestroy(&rx);  VecDestroy(&ry);
    PetscViewerDestroy(&viewer);
  }

  PetscFinalize();
  return 0;
}

