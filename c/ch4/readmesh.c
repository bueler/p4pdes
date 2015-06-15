#include <petscmat.h>
#include <petscksp.h>
#include "readmesh.h"

//STARTGET
PetscErrorCode getmeshfile(MPI_Comm comm, const char suffix[],
                           char filename[], PetscViewer *viewer) {
  PetscErrorCode ierr; //STRIP
  PetscBool      fset;
  ierr = PetscOptionsBegin(comm, "", "options for readmesh", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root with PETSc binary, for reading", "", "",
                      filename, PETSC_MAX_PATH_LEN, &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {  SETERRQ(comm,1,"option  -f FILENAME  required");  }  //STRIP
  strcat(filename,suffix);
  ierr = PetscPrintf(comm,"  opening mesh file %s ...\n",filename); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,viewer); CHKERRQ(ierr);
  return 0;
}
//ENDGET

//STARTREADMESH
PetscErrorCode createloadname(MPI_Comm comm, PetscViewer viewer, const char prefix[],
                              const char name[], Vec *v) {
  PetscErrorCode ierr; //STRIP
  ierr = VecCreate(comm,v); CHKERRQ(ierr);
  ierr = VecSetOptionsPrefix(*v,prefix); CHKERRQ(ierr);
  ierr = VecLoad(*v,viewer); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*v),name); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode getcheckmeshsizes(MPI_Comm comm, Vec E, Vec x, Vec y,
                                 PetscInt *N, PetscInt *K, PetscInt *bs) {
  PetscErrorCode ierr;
  PetscInt Ny;
  if (N) {
    ierr = VecGetSize(x,N); CHKERRQ(ierr);
    ierr = VecGetSize(y,&Ny); CHKERRQ(ierr);
    if (Ny != *N) {  SETERRQ(comm,3,"x,y arrays invalid: must have equal length"); } 
  }
  if (K) {
    ierr = VecGetSize(E,K); CHKERRQ(ierr);
    if (*K % 15 != 0) {  SETERRQ(comm,3,"element array E invalid (!= 15 K entries)"); }
    *K /= 15;
  }
  if (bs) {
    ierr = VecGetBlockSize(E,bs); CHKERRQ(ierr);
    if (*bs != 15) {  SETERRQ(comm,3,"element array E has invalid block size (!= 15)"); }
  }
  return 0;
}

PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer, Vec *E, Vec *x, Vec *y) {
  PetscInt bs,N,K;
  PetscErrorCode ierr; //STRIP
  ierr = PetscPrintf(comm,"  reading mesh Vec E,x,y from file ...\n"); CHKERRQ(ierr);
  ierr = createloadname(comm, viewer, "E_", "E-element-full-info", E); CHKERRQ(ierr);
  ierr = createloadname(comm, viewer, "x_", "x-coordinate", x); CHKERRQ(ierr);
  ierr = createloadname(comm, viewer, "y_", "y-coordinate", y); CHKERRQ(ierr);
  ierr = getcheckmeshsizes(comm,*E,*x,*y,&N,&K,&bs); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"    block size for E is %d\n",bs); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"    N=%d nodes, K=%d elements\n",N,K); CHKERRQ(ierr);
  return 0;
}
//ENDREADMESH


PetscErrorCode showelementSynchronized(MPI_Comm comm, elementtype *et) {
  PetscErrorCode ierr;
  ierr = PetscSynchronizedPrintf(comm,
               "%d %d %d:\n"
               "    %d %d %d | %d %d %d | %g %g %g | %g %g %g |\n",
               (int)et->j[0], (int)et->j[1], (int)et->j[2],
               (int)et->bN[0],(int)et->bN[1],(int)et->bN[2],
               (int)et->bE[0],(int)et->bE[1],(int)et->bE[2],
               et->x[0],      et->x[1],      et->x[2],
               et->y[0],      et->y[1],      et->y[2]); CHKERRQ(ierr);
  return 0;
}


PetscErrorCode elementVecViewSTDOUT(MPI_Comm comm, Vec E) {
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       bs = 15, bsread, k, Kstart, Kend;
  PetscScalar    *ae;
  elementtype    *et;
  MPI_Comm_rank(comm,&rank);
  PetscObjectPrintClassNamePrefixType((PetscObject)E,PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscSynchronizedPrintf(comm, "Process [%d]\n",rank); CHKERRQ(ierr);
  ierr = VecGetBlockSize(E,&bsread); CHKERRQ(ierr);
  if (bsread != bs) {
    SETERRQ1(comm,3,"element array E has invalid block size (!= %d)",bs); }
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k += bs) { // loop over all owned elements
    et = (elementtype*)(&(ae[k-Kstart]));
    ierr = showelementSynchronized(comm, et); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
  return 0;
}
