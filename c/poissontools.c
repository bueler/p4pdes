#include <petscmat.h>
#include "readmesh.h"

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
  PetscInt K, Istart, Iend, Kstart, Kend, bs;
  ierr = getmeshsizes(comm,E,x,y,NULL,&K); CHKERRQ(ierr); // K = # of elements
  ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
  if (15 != bs) {
    SETERRQ1(comm,3,"element node index array E has invalid block size: must be %d",15); }
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
#if DEBUG
  PetscMPIInt    rank;
  MPI_Comm_rank(comm,&rank);
  ierr = PetscPrintf(comm,"    inside prealloc(), on rank %d:  Kstart=%d, Kend=%d\n",
                     rank,Kstart,Kend); CHKERRQ(ierr);
#endif
  for (k = Kstart; k < Kend; k += bs) { // loop over all elements we own
    for (q = 0; q < bs; q++) {  kk[q] = k + q;  }
    ierr = VecGetValues(E,bs,kk,yy); CHKERRQ(ierr);
    et = (elementtype*)(yy);
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
//ENDELEMENTSLOOP

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
  // resolve double counting
  for (iloc = 0; iloc < mm; iloc++) {
    dnnz[iloc] /= 2;
    onnz[iloc] /= 2;
  }

#if DEBUG
  ierr = printnnz(comm, mm, dnnz, onnz); CHKERRQ(ierr);
#endif

  // PREALLOCATE STIFFNESS MATRIX
  ierr = MatMPIAIJSetPreallocation(*A,0,dnnz,0,onnz); CHKERRQ(ierr);
  PetscFree(dnnz);  PetscFree(onnz);
  return 0;
}
//ENDPREALLOC
