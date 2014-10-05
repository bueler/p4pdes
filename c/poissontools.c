#include <petscmat.h>
#include "readmesh.h"

#define DEBUG 0

//STARTPREALLOC
PetscErrorCode prealloc(MPI_Comm comm, Vec E, Vec x, Vec y, Vec Q,
                        PetscInt Istart, PetscInt Iend, Mat *A) {
  PetscErrorCode ierr;
  PetscInt K, M;
  ierr = getmeshsizes(comm,E,x,Q,NULL,&K,&M); CHKERRQ(ierr); // K = # of elements, M = # of bdry segs

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
  PetscScalar *ae;
  elementtype *Eptr;
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  Eptr = (elementtype*)ae;
  for (k = 0; k < K; k++) {          // loop over ALL elements
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      //WAS: i = (int)ap[3*k+q];
      i = (int)Eptr[k].j[q];         //   global index of q node
      if ((i < Istart) || (i >= Iend))  continue; // skip node if I don't own it
      iloc = i - Istart;
      for (r = 0; r < 3; r++) {      // loop over other vertices
        if (r == q)  continue;       // diagonal entry already counted
        j = (int)Eptr[k].j[r];       //   global index of r node
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
  PetscMPIInt     rank;
  MPI_Comm_rank(COMM,&rank);
  ierr = PetscSynchronizedPrintf(COMM,"showing entries of dnnz[%d] on rank %d (DEBUG)\n",
                                 mm,rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(COMM,"dnnz[%d] = %d\n",iloc,dnnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedPrintf(COMM,"showing entries of onnz[%d] on rank %d (DEBUG)\n",
                                 mm,rank); CHKERRQ(ierr);
  for (iloc = 0; iloc < mm; iloc++) {
      ierr = PetscSynchronizedPrintf(COMM,"onnz[%d] = %d\n",iloc,onnz[iloc]); CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(COMM,PETSC_STDOUT); CHKERRQ(ierr);
#endif

  // PREALLOCATE STIFFNESS MATRIX
  ierr = MatMPIAIJSetPreallocation(*A,0,dnnz,0,onnz); CHKERRQ(ierr);
  PetscFree(dnnz);  PetscFree(onnz);
  return 0;
}
//ENDPREALLOC
