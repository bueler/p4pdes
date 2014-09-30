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

PetscErrorCode readmesh(MPI_Comm comm, PetscViewer viewer,
                        PetscInt *N, PetscInt *K, PetscInt *M,
                        Vec *x, Vec *y, Vec *BTseq, Vec *Pseq, Vec *Qseq) {
  PetscErrorCode ierr;
  ierr = PetscPrintf(comm,"  reading x,y,BT,P,Q from file ...\n"); CHKERRQ(ierr);

  // READ IN ARRAYS, AND GET SIZES
  Vec BT, P, Q;
  createloadname(comm, *x, viewer,"node-x-coordinate")
  createloadname(comm, *y, viewer,"node-y-coordinate")
  createloadname(comm, BT, viewer,"node-boundary-type")
  createloadname(comm, P,  viewer,"element-node-indices")
  createloadname(comm, Q,  viewer,"boundary-segment-indices")
  ierr = VecGetSize(*x,N); CHKERRQ(ierr);
  ierr = VecGetSize(P,K); CHKERRQ(ierr);
  ierr = VecGetSize(Q,M); CHKERRQ(ierr);
  if (*K % 3 != 0) {
    SETERRQ(comm,3,"element node index array P invalid: must have 3 K entries"); }
  *K /= 3;
  if (*M % 2 != 0) {
    SETERRQ(comm,3,"element node index array Q invalid: must have 2 M entries"); }
  *M /= 2;
  ierr = PetscPrintf(comm,"  N=%d nodes, K=%d elements, M=%d boundary segments\n",
                     *N,*K,*M); CHKERRQ(ierr);

  // PUT A COPY OF BT,P,Q ON EACH PROCESSOR
  VecScatter  ctx;
  ierr = VecScatterCreateToAll(BT,&ctx,BTseq); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,BT,*BTseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,BT,*BTseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  ierr = VecScatterCreateToAll(P,&ctx,Pseq); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,P,*Pseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,P,*Pseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);
  ierr = VecScatterCreateToAll(Q,&ctx,Qseq); CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,Q,*Qseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,Q,*Qseq,INSERT_VALUES,SCATTER_FORWARD); CHKERRQ(ierr);
  VecScatterDestroy(&ctx);

  VecDestroy(&BT);  VecDestroy(&P);  VecDestroy(&Q);
  return 0;
}
//END
