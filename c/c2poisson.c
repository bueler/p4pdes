
static char help[] =
"Solve the Poisson equation using an unstructured mesh FEM method.\n\
For a one-process, coarse grid example do:\n\
     triangle -pqa1.0 bump   # generates bump.1.{node,ele,poly}\n\
     c2triangle -f bump.1    # reads bump.1.{node,ele,poly} and generates bump.1.petsc\n\
     c2poisson -f bump.1     # reads bump.1.petsc and solves the equation\n\
To see the matrix graphically:\n\
     c2poisson -f bump.1 -a_mat_view draw -draw_pause 5\n\n";

#include <petscksp.h>
#include "convenience.h"
#include "readmesh.h"
#include "poissontools.h"

int main(int argc,char **args) {

  PetscInitialize(&argc,&args,(char*)0,help);
  const MPI_Comm  COMM = PETSC_COMM_WORLD;
  PetscErrorCode  ierr;

  // READ MESH FROM FILE
  Vec      x, y,  // mesh: coords of node
           BT,    // mesh: bdry type,
           P,     //       element index,
           Q;     //       boundary segment index
  PetscInt N,     // number of nodes
           K,     // number of elements
           M;     // number of boundary segments
  char     fname[PETSC_MAX_PATH_LEN];
  PetscViewer viewer;
  ierr = getmeshfile(COMM, fname, &viewer); CHKERRQ(ierr);
  ierr = readmeshseqall(COMM, viewer,
                        &x, &y, &BT, &P, &Q); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = getmeshsizes(COMM,x,P,Q,&N,&K,&M); CHKERRQ(ierr);

  // RELOAD x TO GET OWNERSHIP RANGES, AND ALLOCATE RHS
  Vec xmpi, b;
  PetscInt Istart,Iend;
  ierr = getmeshfile(COMM, fname, &viewer); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  re-reading mesh Vec x to get ownership ranges ...\n"); CHKERRQ(ierr);
  ierr = createload(COMM, viewer, &xmpi); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = VecGetOwnershipRange(xmpi,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecDuplicate(xmpi,&b); CHKERRQ(ierr);
  ierr = VecDestroy(&xmpi); CHKERRQ(ierr);

  // CREATE AND PREALLOCATE MAT
  Mat A;
  ierr = MatCreate(COMM,&A); CHKERRQ(ierr);
  ierr = MatSetType(A,MATMPIAIJ); CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N); CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  preallocating stiffness matrix A ...\n"); CHKERRQ(ierr);
  ierr = prealloc(COMM, x, y, BT, P, Q, Istart, Iend, &A); CHKERRQ(ierr);

  ierr = PetscPrintf(COMM,"  assembling initial stiffness matrix A0 ...\n"); CHKERRQ(ierr);
  
  // ASSEMBLE INITIAL STIFFNESS (IGNORING BOUNDARY VALUES)
  PetscScalar dxi[3]  = {-1.0, 1.0, 0.0}, // grad of basis functions on ref element
              deta[3] = {-1.0, 0.0, 1.0};
  PetscInt    k, q, r, i, ii[3], jj[3];
  PetscBool   ownk;
  PetscScalar *ax, *ay, *ap;
  PetscScalar vv[3], b2, b3, c2, c3, detJ;
  ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
  ierr = VecGetArray(P,&ap); CHKERRQ(ierr);
  for (k = 0; k < K; k++) {          // loop over ALL elements
    // get vertex indices for current element
    ownk = PETSC_FALSE;
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      ii[q] = (int)ap[3*k+q];        //   global index of q node
      if ((ii[q] >= Istart) && (ii[q] < Iend))  ownk = PETSC_TRUE;
    }
    if (!ownk) continue;             // skip elements where we DON'T own any nodes
    // compute element dimension constants, and area; see Elman (1.43)
    b2 = ay[ii[2]] - ay[ii[0]];      // in Elman: = y3 - y1
    b3 = ay[ii[0]] - ay[ii[1]];      //           = y1 - y2
    c2 = ax[ii[0]] - ax[ii[2]];      //           = x1 - x3
    c3 = ax[ii[1]] - ax[ii[0]];      //           = x2 - x1
    detJ = c3 * b2 - c2 * b3;        // note area = fabs(detJ)/2.0
    // compute element stiffness contribution
    for (q = 0; q < 3; q++) {        // loop over vertices of current element
      i = ii[q];                     // row index
      if ((i < Istart) || (i >= Iend))  continue; // skip row if I don't own it
      for (r = 0; r < 3; r++) {      // loop over other vertices
        jj[r] = (int)ap[3*k+r];      //   global index of r node
        vv[r] =  (dxi[q] * b2 + deta[q] * b3) * (dxi[r] * b2 + deta[r] * b3);
        vv[r] += (dxi[q] * c2 + deta[q] * c3) * (dxi[r] * c2 + deta[r] * c3);
        vv[r] /= 2.0 * detJ;
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
  ierr = VecRestoreArray(P,&ap); CHKERRQ(ierr);
  matassembly(A)

  // MINIMAL CHECK IS THAT U=1 IS IN KERNEL
  Vec uone;
  PetscScalar normone, normAone;
  ierr = VecDuplicate(b,&uone); CHKERRQ(ierr);
  ierr = VecSet(uone,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uone,b); CHKERRQ(ierr);             // b = A * uone
  ierr = VecNorm(uone,NORM_2,&normone); CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&normAone); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  check:  |A0 * 1|_2 / |1|_2 = %e   (should be O(eps))\n",
                     normAone/normone); CHKERRQ(ierr);

/* FIXME TO DO second check: compute RHS with f = 1, and
  KSP            ksp;
  MatNullSpace   nullsp;
  MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp); // constants are in null space
  KSPSetNullSpace(ksp, nullsp);
  MatNullSpaceDestroy(&nullsp);
*/

  // CLEAN UP
  MatDestroy(&A);
  VecDestroy(&b);
  VecDestroy(&x);  VecDestroy(&y);
  VecDestroy(&BT);  VecDestroy(&P);  VecDestroy(&Q);
  PetscFinalize();
  return 0;
}
