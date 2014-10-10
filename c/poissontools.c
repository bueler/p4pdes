#include <petscmat.h>
#include "convenience.h"
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


PetscScalar chi(PetscInt q, PetscScalar xi, PetscScalar eta) {
  if (q==1)
    return xi;
  else if (q==2)
    return eta;
  else
    return 1.0 - xi - eta;
}


PetscErrorCode initassemble(MPI_Comm comm,
                            Vec E,         // array of elementtype, as read by readmesh()
                            PetscScalar (*f)(PetscScalar, PetscScalar),
                            PetscScalar (*gamma)(PetscScalar, PetscScalar),
                            Mat A, Vec b) {
  PetscErrorCode ierr;  //STRIP
  PetscScalar dxi[3]  = {-1.0, 1.0, 0.0},   // grad of basis functions chi0, chi1, chi2
              deta[3] = {-1.0, 0.0, 1.0},   //     on ref element
              quadxi[3]  = {0.5, 0.5, 0.0}, // quadrature points are midpoints of
              quadeta[3] = {0.0, 0.5, 0.5}; //     sides of ref element
  PetscInt    bs, Kstart, Kend, k, q, r, i, jj[3];
  PetscScalar *ae;
  elementtype *et;
  PetscScalar vv[3], y20, x02, y01, x10, detJ,
              bval, xquad, yquad;
  ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k += bs) {    // loop through owned elements
    et = (elementtype*)(&(ae[k-Kstart]));  // points to current element
    // compute element geometry constants (compare Elman (1.43)
    y20 = et->y[2] - et->y[0];
    x02 = et->x[0] - et->x[2];
    y01 = et->y[0] - et->y[1];
    x10 = et->x[1] - et->x[0];
    detJ = x10 * y20 - y01 * x02;          // note area = fabs(detJ)/2.0
    // loop over vertices of current element
    for (q = 0; q < 3; q++) {
      // compute element stiffness contributions
      i = (int)et->j[q];                   // global row index
      for (r = 0; r < 3; r++) {            // loop over other vertices
        jj[r] = (int)et->j[r];             // global column index
        vv[r] =  (dxi[q] * y20 + deta[q] * y01) * (dxi[r] * y20 + deta[r] * y01);
        vv[r] += (dxi[q] * x02 + deta[q] * x10) * (dxi[r] * x02 + deta[r] * x10);
        vv[r] /= 2.0 * detJ;
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
      // compute element RHS contribution FIXME in homogeneous Neumann case
      bval = 0.0;
      for (r = 0; r < 3; r++) {      // loop over quadrature points
        xquad = et->x[0] + x10 * quadxi[r] - x02 * quadeta[r]; // = x0 + (x1-x0) xi + (x2-x0) eta
        yquad = et->x[0] - y01 * quadxi[r] + y20 * quadeta[r]; // = y0 + (y1-y0) xi + (y2-y0) eta
        bval += (*f)(xquad,yquad) * chi(q,quadxi[r],quadeta[r]);
      }
      bval *= detJ / 6.0;
      ierr = VecSetValues(b,1,&i,&bval,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  matassembly(A)
  vecassembly(b)
  return 0;
}
