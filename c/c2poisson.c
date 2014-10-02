
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

PetscScalar chi(PetscInt q, PetscScalar xi, PetscScalar eta) {
  if (q==0) {
    return 1.0 - xi - eta;
  } else if (q==1) {
    return xi;
  } else {
    return eta;
  }
}

PetscScalar exactsolution(PetscScalar x, PetscScalar y) {
  return cos(2.0*PETSC_PI*x) * cos(2.0*PETSC_PI*y);
}

PetscScalar sourcefunction(PetscScalar x, PetscScalar y) {
  return 8.0 * PETSC_PI * PETSC_PI * exactsolution(x,y);
}

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

  // RELOAD x TO GET OWNERSHIP RANGES, AND ALLOCATE RHS b
  Vec xmpi, b;
  PetscInt Istart,Iend;
  ierr = getmeshfile(COMM, fname, &viewer); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  re-reading mesh Vec x to get ownership ranges ...\n"); CHKERRQ(ierr);
  ierr = createload(COMM, viewer, &xmpi); CHKERRQ(ierr);
  PetscViewerDestroy(&viewer);
  ierr = VecGetOwnershipRange(xmpi,&Istart,&Iend); CHKERRQ(ierr);
  ierr = VecDuplicate(xmpi,&b); CHKERRQ(ierr);
  ierr = VecSet(b,0.0); CHKERRQ(ierr);
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
              deta[3] = {-1.0, 0.0, 1.0},
              quadxi[3]  = {0.5, 0.5, 0.0}, // quadrature points are midpoints of
              quadeta[3] = {0.0, 0.5, 0.5}; //   sides of ref element
  PetscInt    k, q, r, i, ii[3], jj[3];
  PetscBool   ownk;
  PetscScalar *ax, *ay, *ap, *ab;
  PetscScalar vv[3], b2, b3, c2, c3, detJ;
  PetscScalar bval, xquad, yquad;
  ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
  ierr = VecGetArray(P,&ap); CHKERRQ(ierr);
  ierr = VecGetArray(b,&ab); CHKERRQ(ierr);
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
    // loop over vertices of current element
    for (q = 0; q < 3; q++) {
      // compute element stiffness contribution
      i = ii[q];                     // row index
      if ((i < Istart) || (i >= Iend))  continue; // skip row if I don't own it
      for (r = 0; r < 3; r++) {      // loop over other vertices
        jj[r] = (int)ap[3*k+r];      //   global index of r node
        vv[r] =  (dxi[q] * b2 + deta[q] * b3) * (dxi[r] * b2 + deta[r] * b3);
        vv[r] += (dxi[q] * c2 + deta[q] * c3) * (dxi[r] * c2 + deta[r] * c3);
        vv[r] /= 2.0 * detJ;
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
      // compute element RHS contribution
      bval = 0.0;
      for (r = 0; r < 3; r++) {      // loop over quadrature points
        xquad = ax[i] + c3 * quadxi[r] - c2 * quadeta[r]; // = x1 + (x2-x1) xi + (x3 - x1) eta
        yquad = ay[i] - b3 * quadxi[r] + b2 * quadeta[r]; // = y1 + (y2-y1) xi + (y3 - y1) eta
        bval += sourcefunction(xquad,yquad) * chi(q,quadxi[r],quadeta[r]);
      }
      bval *= detJ / 6.0;
      ierr = VecSetValues(b,1,&i,&bval,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
  ierr = VecRestoreArray(P,&ap); CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&ab); CHKERRQ(ierr);
  // ACTUALLY ASSEMBLE
  matassembly(A)
  vecassembly(b)

  // MINIMAL CHECK IS THAT U=1 IS IN KERNEL
  Vec uone, btest;
  PetscScalar normone, normAone;
  ierr = VecDuplicate(b,&uone); CHKERRQ(ierr);
  ierr = VecDuplicate(b,&btest); CHKERRQ(ierr);
  ierr = VecSet(uone,1.0); CHKERRQ(ierr);
  ierr = MatMult(A,uone,btest); CHKERRQ(ierr);             // btest = A * uone
  ierr = VecNorm(uone,NORM_2,&normone); CHKERRQ(ierr);
  ierr = VecNorm(btest,NORM_2,&normAone); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  check I:  |A0 * 1|_2 / |1|_2 = %e   (should be O(eps))\n",
                     normAone/normone); CHKERRQ(ierr);
  ierr = VecDestroy(&uone); CHKERRQ(ierr);
  ierr = VecDestroy(&btest); CHKERRQ(ierr);

  // SECOND CHECK: SOLVE HOMOGENEOUS NEUMANN PROBLEM WITH KNOWN SOLN
  // FIRST EVALUATE EXACT SOLUTION AT NODES
  Vec         uexact;
  PetscScalar uval;
  ierr = VecDuplicate(b,&uexact); CHKERRQ(ierr);
  ierr = VecGetArray(x,&ax); CHKERRQ(ierr);
  ierr = VecGetArray(y,&ay); CHKERRQ(ierr);
  for (i = Istart; i < Iend; i++) {
    uval = exactsolution(ax[i],ay[i]);
    ierr = VecSetValues(uexact,1,&i,&uval,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&ax); CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&ay); CHKERRQ(ierr);
  vecassembly(uexact)
  // NEXT SOLVE SYSTEM
  Vec          u;
  KSP          ksp;
  MatNullSpace nullsp;
  ierr = VecDuplicate(b, &u); CHKERRQ(ierr);
  ierr = KSPCreate(COMM, &ksp); CHKERRQ(ierr);
  // constants are in null space:
  ierr = MatNullSpaceCreate(COMM, PETSC_TRUE, 0, NULL, &nullsp); CHKERRQ(ierr);
  ierr = KSPSetNullSpace(ksp, nullsp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, u); CHKERRQ(ierr);
  // NOW COMPUTE ERROR
  PetscScalar normuexact, normerror;
  ierr = VecNorm(uexact,NORM_2,&normuexact); CHKERRQ(ierr);
  ierr = VecAXPY(u,-1.0,uexact); CHKERRQ(ierr);  // u := -uexact + u
  ierr = VecNorm(u,NORM_2,&normerror); CHKERRQ(ierr);
  ierr = PetscPrintf(COMM,"  check II: |u - uexact|_2 / |uexact|_2 = %e  (should be O(h^2))\n",
                     normerror/normuexact); CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  // CLEAN UP
  MatDestroy(&A);
  VecDestroy(&b);  VecDestroy(&u);  VecDestroy(&uexact);
  VecDestroy(&x);  VecDestroy(&y);
  VecDestroy(&BT);  VecDestroy(&P);  VecDestroy(&Q);
  PetscFinalize();
  return 0;
}
