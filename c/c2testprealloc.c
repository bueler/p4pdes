
static char help[] =
"Read in a FEM grid (unstructured triangulation) from PETSc binary file.\n\
Demonstrate Mat preallocation.\n\
For a one-process, coarse grid example do:\n\
     triangle -pqa1.0 bump       # generates bump.1.{node,ele,poly}\n\
     c2triangle -f bump.1        # reads bump.1.{node,ele,poly} and generates bump.1.petsc\n\
     c2testprealloc -f bump.1    # reads bump.1.petsc\n\
To see the sparsity pattern graphically:\n\
     c2testprealloc -f bump.1 -a_mat_view draw -draw_pause 5\n\n";

#include <petscmat.h>
#include "convenience.h"
#include "readmesh.h"
#include "poissontools.h"
#define DEBUG 0

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;

//STARTLOAD
  // READ MESH FROM FILE
  Vec      x, y,  // mesh: coords of node
           BT,    // mesh: bdry type,
           P,     //       element index,
           Q;     //       boundary segment index
  PetscInt N,     // number of nodes
           K;     // number of elements
  char     fname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  ierr = getmeshfile(COMM, fname, &viewer); CHKERRQ(ierr);
  ierr = readmeshseqall(COMM, viewer,
                        &x, &y, &BT, &P, &Q); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = VecGetSize(x,&N); CHKERRQ(ierr);
  ierr = VecGetSize(P,&K); CHKERRQ(ierr);
  if (K % 3 != 0) {
    SETERRQ(COMM,3,"element node index array P invalid: must have 3 K entries"); }
  K /= 3;

  // RELOAD x TO GET OWNERSHIP RANGES
  Vec xmpi;
  PetscInt Istart,Iend;
  ierr = getmeshfile(COMM, fname, &viewer); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  re-reading mesh Vec x to get ownership ranges ...\n"); CHKERRQ(ierr);
  ierr = createload(COMM, viewer, &xmpi); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = VecGetOwnershipRange(xmpi,&Istart,&Iend); CHKERRQ(ierr);
//ENDLOAD

  // CREATE AND PREALLOCATE MAT
  Mat A;
  ierr = MatCreate(COMM,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  preallocating stiffness matrix A ...\n"); CHKERRQ(ierr);
  ierr = prealloc(COMM, x, y, BT, P, Q, Istart, Iend, &A); CHKERRQ(ierr);

  // FILL MAT WITH FAKE ENTRIES
  PetscInt    k, q, r, i, jj[3];
  PetscScalar *ap, vv[3];
  ierr = VecGetArray(P,&ap); CHKERRQ(ierr);
  for (k = 0; k < K; k++) {          // loop over ALL elements
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = (int)ap[3*k+q];            //   global index of q node
      if ((i < Istart) || (i >= Iend))  continue; // skip node if I don't own it
      for (r = 0; r < 3; r++) {      // loop over other vertices
        jj[r] = (int)ap[3*k+r];      //   global index of r node
        vv[r] = 1.0;
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(P,&ap); CHKERRQ(ierr);
  matassembly(A)
//ENDTEST

  // CLEAN UP
  MatDestroy(&A);
  VecDestroy(&x);  VecDestroy(&y);
  VecDestroy(&BT);  VecDestroy(&P);  VecDestroy(&Q);
  PetscFinalize();
  return 0;
}
