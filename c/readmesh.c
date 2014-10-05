#include <petscmat.h>
#include <petscksp.h>
#include "convenience.h"
#include "readmesh.h"

//STARTGET
PetscErrorCode getmeshfile(MPI_Comm comm, const char suffix[],
                           char filename[], PetscViewer *viewer) {
  PetscErrorCode ierr;
  PetscBool      fset;
  ierr = PetscOptionsBegin(comm, "", "options for readmesh", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root with PETSc binary, for reading", "", "",
                            filename, PETSC_MAX_PATH_LEN, &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(comm,1,"option  -f FILENAME  required");
  }
  strcat(filename,suffix);
  ierr = PetscPrintf(comm,"  opening mesh file %s ...\n",filename); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,
             viewer); CHKERRQ(ierr);
  return 0;
}
//ENDGET

//STARTREADMESH
PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer,
                        Vec *E, Vec *x, Vec *y) {
  PetscInt bs,N,K;
  PetscErrorCode ierr;
  ierr = PetscPrintf(comm,"  reading mesh Vec E,x,y from file ...\n"); CHKERRQ(ierr);
  ierr = VecCreate(comm,E); CHKERRQ(ierr);
  VecSetOptionsPrefix(*E,"E_");
  ierr = VecLoad(*E,viewer); CHKERRQ(ierr);
  ierr = VecCreate(comm,x); CHKERRQ(ierr);
  VecSetOptionsPrefix(*x,"x_");
  ierr = VecLoad(*x,viewer); CHKERRQ(ierr);
  ierr = VecCreate(comm,y); CHKERRQ(ierr);
  VecSetOptionsPrefix(*y,"y_");
  ierr = VecLoad(*y,viewer); CHKERRQ(ierr);
  ierr = VecGetBlockSize(*E,&bs); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"    block size for E is %d\n",bs); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*E),"E-element-full-info"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*x),"x-coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*y),"y-coordinate"); CHKERRQ(ierr);
  ierr = getmeshsizes(comm,*E,*x,*y,&N,&K); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"    N=%d nodes, K=%d elements\n",
                     N,K); CHKERRQ(ierr);
  return 0;
}
//ENDREADMESH

PetscErrorCode getmeshsizes(MPI_Comm comm, Vec E, Vec x, Vec y,
                            PetscInt *N, PetscInt *K) {
  PetscErrorCode ierr;
  PetscInt Ny, bs;
  if (N) {
    ierr = VecGetSize(x,N); CHKERRQ(ierr);
    ierr = VecGetSize(y,&Ny); CHKERRQ(ierr);
    if (Ny != *N) {
      SETERRQ(comm,3,"x,y-coordinate arrays invalid: must have equal length"); } 
  }
  if (K) {
    ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
    if (bs != 15) {
      SETERRQ(comm,3,"element node index array E has invalid block size: must be 15"); }
    ierr = VecGetSize(E,K); CHKERRQ(ierr);
    if (*K % 15 != 0) {
      SETERRQ(comm,3,"element node index array E invalid: must have 15 K entries"); }
    *K /= 15;
  }
  return 0;
}

// FIXME: for some reason this is not giving the view I want in parallel
PetscErrorCode elementVecViewSTDOUT(MPI_Comm comm, Vec E) {
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       bs, k, Kstart,Kend;
  PetscScalar    *ae;
  elementtype    *et;
  MPI_Comm_rank(comm,&rank);
  PetscObjectPrintClassNamePrefixType((PetscObject)E,PETSC_VIEWER_STDOUT_WORLD);
  ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "Process [%d]\n",rank); CHKERRQ(ierr);
  ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
  if (bs != 15) {
    SETERRQ(comm,3,"element node index array E has invalid block size: must be 15"); }
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);
  ierr = PetscSynchronizedPrintf(comm, "  Kstart = %d, Kend = %d\n",Kstart,Kend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k = k+bs) { // loop over all owned elements
    et = (elementtype*)(&(ae[k]));
    ierr = PetscSynchronizedPrintf(comm,
               "%d %d %d:\n"
               "    %d %d %d | %d %d %d | %f %f %f | %f %f %f|\n",
               (int)et->j[0], (int)et->j[1], (int)et->j[2],
               (int)et->bN[0],(int)et->bN[1],(int)et->bN[2],
               (int)et->bE[0],(int)et->bE[1],(int)et->bE[2],
               et->x[0],      et->x[1],      et->x[2],
               et->y[0],      et->y[1],      et->y[2]); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
  return 0;
}
