// ABANDONED CODE FOR PARALLEL PREALLOCATION

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
#include "ch4/convenience.h"
#include "ch4/readmesh.h"

#define DEBUG 1

PetscErrorCode printnnz(MPI_Comm comm, PetscInt mm, PetscInt *dnnz, PetscInt *onnz) {
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       iloc;
  MPI_Comm_rank(comm,&rank);
  ierr = PetscSynchronizedPrintf(comm,"showing entries of dnnz[%d] on rank %d (DEBUG)\n",
                                 mm,rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(comm,"dnnz[%d] = %d\n",iloc,dnnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedPrintf(comm,"showing entries of onnz[%d] on rank %d (DEBUG)\n",
                                 mm,rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(comm,"onnz[%d] = %d\n",iloc,onnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT); CHKERRQ(ierr);
  return 0;
}

//STARTPREALLOC
PetscErrorCode prealloc(MPI_Comm comm, Vec E, Vec x, Vec y, Mat A) {
  PetscErrorCode ierr;
  PetscInt K, bs, Istart, Iend, Kstart, Kend;
  ierr = getcheckmeshsizes(comm,E,x,y,NULL,&K,&bs); CHKERRQ(ierr); // K = # of elements
  ierr = VecGetOwnershipRange(x,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);

  // ALLOCATE LOCAL ARRAYS FOR NUMBER OF NONZEROS
  PetscInt mm = Iend - Istart, iloc;
  int *dnnz, // dnnz[i] is number of nonzeros in row which are in same-processor column
      *onnz; // onnz[i] is number of nonzeros in row which are in other-processor column
  PetscMalloc(mm*sizeof(int),&dnnz);
  PetscMalloc(mm*sizeof(int),&onnz);
  for (iloc = 0; iloc < mm; iloc++) {
    dnnz[iloc] = 2;  // diagonal entry
    onnz[iloc] = 0;
  }

  // FILL THE NUMBER-OF-NONZEROS ARRAYS: LOOP OVER ELEMENTS
  PetscInt    i, j, k, q, r;
  elementtype *et;
  PetscScalar *ae;
#if DEBUG
  PetscMPIInt    rank;
  MPI_Comm_rank(comm,&rank);
  ierr = PetscPrintf(comm,"    inside prealloc(), on rank %d:  Kstart=%d, Kend=%d\n",
                     rank,Kstart,Kend); CHKERRQ(ierr);
#endif
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k += bs) { // loop over all elements we own
    et = (elementtype*)(&(ae[k-Kstart]));
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = (int)(et->j[q]);           //   global index of q node
      if ((i < Istart) || (i >= Iend))  continue; // skip node if I don't own it
      iloc = i - Istart;
      for (r = 0; r < 3; r++) {      // loop over other vertices
        if (r == q)  continue;       // diagonal entry already counted
        j = (int)(et->j[r]);         //   global index of q node
        // (i,j) is an edge; we count this nonzero matrix entry
        if ((j >= Istart) && (j < Iend)) {
          dnnz[iloc]++;
        } else {
          onnz[iloc]++;
        }
      }
    }
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
//ENDELEMENTSLOOP

#if 0
FIXME:  this part needs a replacement based on looping over E
  // FILL THE NUMBER-OF-NONZEROS ARRAYS: LOOP OVER BOUNDARY SEGMENTS
  PetscInt    m;
  PetscScalar *aq;
  ierr = VecGetArray(Q,&aq); CHKERRQ(ierr);
  for (m = 0; m < M; m++) {          // loop over ALL boundary segments
    for (q = 0; q < 2; q++) {        // loop over vertices of current segment
      i = (int)aq[2*m+q];            //   global index of q node
      if ((i < Istart) || (i >= Iend))  continue; // skip node if I don't own it
      iloc = i - Istart;
      r = 1 - q;                     // other vertex
      j = (int)aq[2*m+r];            //   global index of r node
      // (i,j) is a boundary segment; we count this nonzero matrix entry AGAIN
      if ((j >= Istart) && (j < Iend)) {
        dnnz[iloc]++;
      } else {
        onnz[iloc]++;
      }
    }
  }
  ierr = VecRestoreArray(Q,&aq); CHKERRQ(ierr);
#endif

  // resolve double counting
  for (iloc = 0; iloc < mm; iloc++) {
    dnnz[iloc] /= 2;
    onnz[iloc] /= 2;
  }

#if DEBUG
  ierr = printnnz(comm, mm, dnnz, onnz); CHKERRQ(ierr);
#endif

  // PREALLOCATE STIFFNESS MATRIX
  ierr = MatMPIAIJSetPreallocation(A,0,dnnz,0,onnz); CHKERRQ(ierr);
  PetscFree(dnnz);  PetscFree(onnz);
  return 0;
}
//ENDPREALLOC



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
  ierr = getmeshsizes(COMM, E, x, y, &N, &K); CHKERRQ(ierr);
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
