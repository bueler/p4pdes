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
//ENDGET

PetscErrorCode createload(MPI_Comm comm, PetscViewer viewer, Vec *X) {
  PetscErrorCode ierr;
  ierr = VecCreate(comm,X); CHKERRQ(ierr);
  ierr = VecLoad(*X,viewer); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode readmeshseqall(MPI_Comm comm, PetscViewer viewer,
                              PetscInt *N, PetscInt *K, PetscInt *M,
                              Vec *x, Vec *y, Vec *BT, Vec *P, Vec *Q) {
  PetscErrorCode ierr;
  ierr = PetscPrintf(comm,"  reading Vecs x,y,BT,P,Q from file ...\n"); CHKERRQ(ierr);

  // READ IN ARRAYS, AND GET SIZES
  Vec xmpi, ympi, BTmpi, Pmpi, Qmpi;
  ierr = createload(comm, viewer, &xmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &ympi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &BTmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &Pmpi); CHKERRQ(ierr);
  ierr = createload(comm, viewer, &Qmpi); CHKERRQ(ierr);

  ierr = VecGetSize(xmpi,N); CHKERRQ(ierr);
  ierr = VecGetSize(Pmpi,K); CHKERRQ(ierr);
  ierr = VecGetSize(Qmpi,M); CHKERRQ(ierr);
  if (*K % 3 != 0) {
    SETERRQ(comm,3,"element node index array P invalid: must have 3 K entries"); }
  *K /= 3;
  if (*M % 2 != 0) {
    SETERRQ(comm,3,"element node index array Q invalid: must have 2 M entries"); }
  *M /= 2;
  ierr = PetscPrintf(comm,"    N=%d nodes, K=%d elements, M=%d boundary segments\n",
                     *N,*K,*M); CHKERRQ(ierr);

  // COPY TO EACH PROCESSOR
  ierr = PetscPrintf(comm,"  scattering each Vec to each processor ...\n"); CHKERRQ(ierr);
  VecScatter  ctx;
  // scatter N-length Vecs
  ierr = VecScatterCreateToAll(xmpi,&ctx,x); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,xmpi,*x,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,xmpi,*x,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  ierr = VecScatterCreateToAll(ympi,&ctx,y); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,ympi,*y,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,ympi,*y,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  ierr = VecScatterCreateToAll(BTmpi,&ctx,BT); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,BTmpi,*BT,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,BTmpi,*BT,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  // scatter 3K-length Vec
  ierr = VecScatterCreateToAll(Pmpi,&ctx,P); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,Pmpi,*P,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,Pmpi,*P,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  // scatter 2M-length Vec
  ierr = VecScatterCreateToAll(Qmpi,&ctx,Q); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,Qmpi,*Q,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,Qmpi,*Q,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
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
