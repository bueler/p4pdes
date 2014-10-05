
static char help[] =
"Read in a FEM grid (unstructured triangulation) from PETSc binary file.\n\
Demonstrate Mat preallocation.\n\
For a one-process, coarse grid example do:\n\
     triangle -pqa1.0 bump     # generates bump.1.{node,ele,poly}\n\
     c2convert -f bump.1       # reads bump.1.{node,ele,poly}; generate bump.1.petsc\n\
     c2testprealloc -f bump.1  # reads bump.1.petsc and tests preallocation\n\
To see the sparsity pattern graphically:\n\
     c2testprealloc -f bump.1 -a_mat_view draw -draw_pause 5\n\n";

#include <petscmat.h>
#include "convenience.h"
#include "readmesh.h"
#include "poissontools.h"

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;

//STARTLOAD
  Vec      E,     // full element info
           x, y,  // coords of node
           Q;     // boundary segment index
  PetscInt N,     // number of nodes = number of rows
           K;     // number of elements
  char     fname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  ierr = getmeshfile(COMM, ".petsc", fname, &viewer); CHKERRQ(ierr);
  ierr = readmesh(COMM, viewer,
                  &E, &x, &y, &Q); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = getmeshsizes(COMM, E, x, Q, &N, &K, NULL); CHKERRQ(ierr);
//ENDLOAD

  // CREATE AND PREALLOCATE MAT
  Mat A;
  ierr = MatCreate(COMM,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  preallocating stiffness matrix A ...\n"); CHKERRQ(ierr);
  ierr = prealloc(COMM, E, x, y, Q, &A); CHKERRQ(ierr);

  // FILL MAT WITH FAKE ENTRIES
  PetscInt    k, q, r, i, jj[3];
  PetscInt    Istart,Iend;
  PetscScalar *ae, vv[3];
  elementtype *Eptr;
  ierr = VecGetOwnershipRange(x,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  Eptr = (elementtype*)ae;
  for (k = 0; k < K; k++) {          // loop over ALL elements
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = (int)Eptr[k].j[q];         //   global index of q node
      if ((i < Istart) || (i >= Iend))  continue; // skip node if I don't own it
      for (r = 0; r < 3; r++) {      // loop over other vertices
        jj[r] = (int)Eptr[k].j[r];   //   global index of r node
        vv[r] = 1.0;
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  matassembly(A)
//ENDTEST

  // CLEAN UP
  MatDestroy(&A);
  VecDestroy(&E);  VecDestroy(&x);  VecDestroy(&y);  VecDestroy(&Q);
  PetscFinalize();
  return 0;
}
