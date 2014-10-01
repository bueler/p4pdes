#include <petscmat.h>
#include <petscksp.h>
#include "convenience.h"
#include "readmesh.h"

//STARTGET
PetscErrorCode getmeshfile(MPI_Comm comm, char filename[], PetscViewer *viewer) {
  PetscErrorCode ierr;
  PetscBool      fset;
  ierr = PetscOptionsBegin(comm, "", "options for readmesh", ""); CHKERRQ(ierr);
  ierr = PetscOptionsString("-f", "filename root with PETSc binary, for reading", "", "",
                            filename, sizeof(filename), &fset); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);
  if (!fset) {
    SETERRQ(comm,1,"option  -f FILENAME  required");
  }
  strcat(filename,".petsc");
  ierr = PetscPrintf(comm,"  opening mesh file %s ...\n",filename); CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,
             viewer); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode createload(MPI_Comm comm, PetscViewer viewer, Vec *X) {
  PetscErrorCode ierr;
  ierr = VecCreate(comm,X); CHKERRQ(ierr);
  ierr = VecLoad(*X,viewer); CHKERRQ(ierr);
  return 0;
}
//ENDGET

PetscErrorCode readmeshseqall(MPI_Comm comm, PetscViewer viewer,
                              Vec *x, Vec *y, Vec *BT, Vec *P, Vec *Q) {
  PetscErrorCode ierr;
  ierr = PetscPrintf(comm,"  reading mesh Vecs x,y,BT,P,Q from file ...\n"); CHKERRQ(ierr);

  // READ IN ARRAYS, AND GET SIZES (SANITY CHECK)
  Vec xmpi, ympi, BTmpi, Pmpi, Qmpi;
  PetscInt N,K,M;
  ierr = createload(comm, viewer, &xmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &ympi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &BTmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &Pmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &Qmpi); CHKERRQ(ierr);
  ierr = getmeshsizes(comm,xmpi,Pmpi,Qmpi,&N,&K,&M); CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"    N=%d nodes, K=%d elements, M=%d boundary segments\n",
                     N,K,M); CHKERRQ(ierr);

  // COPY TO EACH PROCESSOR
  ierr = PetscPrintf(comm,"  scattering each mesh Vec to each process ...\n"); CHKERRQ(ierr);
  VecScatter  ctx;
  scatterforwardall(ctx,xmpi,*x)
  scatterforwardall(ctx,ympi,*y)
  scatterforwardall(ctx,BTmpi,*BT)
  scatterforwardall(ctx,Pmpi,*P)
  scatterforwardall(ctx,Qmpi,*Q)
  VecDestroy(&xmpi);  VecDestroy(&ympi);
  VecDestroy(&BTmpi);  VecDestroy(&Pmpi);  VecDestroy(&Qmpi);

  ierr = PetscObjectSetName((PetscObject)(*x),"node-x-coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*y),"node-y-coordinate"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*BT),"node-boundary-type"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*P),"element-node-indices"); CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)(*Q),"boundary-segment-indices"); CHKERRQ(ierr);
  return 0;
}
//END

PetscErrorCode getmeshsizes(MPI_Comm comm, Vec x, Vec P, Vec Q,
                            PetscInt *N, PetscInt *K, PetscInt *M) {
  PetscErrorCode ierr;
  if (N) {  ierr = VecGetSize(x,N); CHKERRQ(ierr); }
  if (K) {
    ierr = VecGetSize(P,K); CHKERRQ(ierr);
    if (*K % 3 != 0) {
      SETERRQ(comm,3,"element node index array P invalid: must have 3 K entries"); }
    *K /= 3;
  }
  if (M) {
    ierr = VecGetSize(Q,M); CHKERRQ(ierr);
    if (*M % 2 != 0) {
      SETERRQ(comm,3,"element node index array Q invalid: must have 2 M entries"); }
    *M /= 2;
  }
  return 0;
}
