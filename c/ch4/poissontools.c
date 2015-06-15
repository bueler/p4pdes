#include <petscmat.h>
#include "readmesh.h"


PetscScalar chi(PetscInt q, PetscScalar xi, PetscScalar eta) {
  if (q==1)
    return xi;
  else if (q==2)
    return eta;
  else
    return 1.0 - xi - eta;
}


//DIRICHLETROWS
PetscErrorCode dirichletrows(MPI_Comm comm,
                             Vec E,
                             PetscScalar (*g)(PetscScalar, PetscScalar),
                             Mat A, Vec b) {
  PetscErrorCode ierr;  //STRIP
  PetscInt       bs, Kstart, Kend, k, i, q;
  PetscScalar    *ae, one=1.0, bi;
  elementtype    *et;
  ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k += bs) {    // loop through owned elements
    et = (elementtype*)(&(ae[k-Kstart]));  // points to current element
    // loop over vertices of current element
    for (q = 0; q < 3; q++) {
      if ((int)et->bN[q] == 2) {           // node is Dirichlet
        i = (int)et->j[q];                 // global row index
        ierr = MatSetValues(A,1,&i,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
        bi = (*g)(et->x[q],et->y[q]);
        ierr = VecSetValues(b,1,&i,&bi,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  return 0;
}
//ENDDIRICHLETROWS


//ASSEMBLEADDONE
PetscErrorCode assembleadd(MPI_Comm comm,
                           Vec E,
                           PetscScalar (*f)(PetscScalar, PetscScalar),
                           PetscScalar (*g)(PetscScalar, PetscScalar),
                           PetscScalar (*gamma)(PetscScalar, PetscScalar),
                           Mat A, Vec b) {
  PetscErrorCode ierr;  //STRIP
  PetscScalar dxi[3]  = {-1.0, 1.0, 0.0},   // grad of basis functions chi0, chi1, chi2
              deta[3] = {-1.0, 0.0, 1.0},   //     on ref element
              quadxi[3]  = {0.5, 0.5, 0.0}, // quadrature points are midpoints of
              quadeta[3] = {0.0, 0.5, 0.5}; //     sides of ref element
  PetscInt    bs, Kstart, Kend, k, q, r, rnext, i, jj[3];
  PetscScalar *ae;
  elementtype *et;
  PetscScalar y20, x02, y01, x10, detJ, vv[3],
              bval, xquad[3], yquad[3], slen;
  ierr = VecGetBlockSize(E,&bs); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(E,&Kstart,&Kend); CHKERRQ(ierr);
  ierr = VecGetArray(E,&ae); CHKERRQ(ierr);
  for (k = Kstart; k < Kend; k += bs) {    // loop through owned elements
    et = (elementtype*)(&(ae[k-Kstart]));  // points to current element
    // compute coordinate differences (compare Elman (1.43)
    x10 = et->x[1] - et->x[0];  x02 = et->x[0] - et->x[2];
    y01 = et->y[0] - et->y[1];  y20 = et->y[2] - et->y[0];
    detJ = x10 * y20 - y01 * x02;          // note area = fabs(detJ)/2.0
    // store element quadrature points (x,y coordinates)
    for (r = 0; r < 3; r++) {
      // recall x = x0 + (x1-x0) xi + (x2-x0) eta  //STRIP
      //   and  y = y0 + (y1-y0) xi + (y2-y0) eta  //STRIP
      xquad[r] = et->x[0] + x10 * quadxi[r] - x02 * quadeta[r];
      yquad[r] = et->x[0] - y01 * quadxi[r] + y20 * quadeta[r];
    }
//ENDONE
    // loop over vertices of current element
    for (q = 0; q < 3; q++) {
      if ( (g) && ((int)et->bN[q] == 2) )  continue;  // skip Dirichlet rows
      i = (int)et->j[q];                   // global row index
      // compute element RHS contribution from source f
      bval = 0.0;
      if (f) {
        for (r = 0; r < 3; r++) {          // loop over quadrature points
          bval += (*f)(xquad[r],yquad[r]) * chi(q,quadxi[r],quadeta[r]);
        }
        bval *= fabs(detJ) / 6.0;
      }
      // add element RHS contribution from Neumann boundary values gamma
      if (gamma) {
        for (r = 0; r < 3; r++) {          // loop over edges of elemtn
          if ((int)et->bE[r]) {            // segment is on boundary
            rnext = (r < 2) ? r+1 : 0;     // cycle
            if ( ((int)et->bN[r] == 3) && ((int)et->bN[rnext] == 3) ) {
              // both end nodes must be Neumann to be Neumann boundary
              slen = PetscSqr(et->x[rnext] - et->x[r]) + PetscSqr(et->y[rnext] - et->y[r]);
              slen = PetscSqrtReal(slen);  // = side (edge) length
              bval += (*gamma)(xquad[r],yquad[r]) * chi(q,quadxi[r],quadeta[r]) * slen;
            }
          }
        }
      }
      // compute initial element stiffness contributions (ignore Dirchlet)
      for (r = 0; r < 3; r++) {            // loop over other vertices
        jj[r] = (int)et->j[r];             // global column index
        vv[r] =  (dxi[q] * y20 + deta[q] * y01) * (dxi[r] * y20 + deta[r] * y01);
        vv[r] += (dxi[q] * x02 + deta[q] * x10) * (dxi[r] * x02 + deta[r] * x10);
        vv[r] /= 2.0 * fabs(detJ);
      }
      // edit Dirichlet columns, and finalize right-hand side
      if (g) {
        for (r = 0; r < 3; r++) {          // loop over other vertices
          if ((int)et->bN[r] == 2) {
            bval -= (*g)(et->x[r],et->y[r]) * vv[r];
            vv[r] = 0.0;
          }
        }
      }
      ierr = MatSetValues(A,1,&i,3,jj,vv,ADD_VALUES); CHKERRQ(ierr);
      ierr = VecSetValues(b,1,&i,&bval,ADD_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(E,&ae); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  return 0;
}
//ENDASSEMBLEADD

//FULLASSEMBLE
PetscErrorCode assemble(MPI_Comm comm,
                        Vec E,
                        PetscScalar (*f)(PetscScalar, PetscScalar),
                        PetscScalar (*g)(PetscScalar, PetscScalar),
                        PetscScalar (*gamma)(PetscScalar, PetscScalar),
                        Mat A, Vec b) {
  PetscErrorCode ierr;  //STRIP
  if (g) {
    ierr = dirichletrows(comm,E,g,A,b); CHKERRQ(ierr);
  }
  ierr = assembleadd(comm,E,f,g,gamma,A,b); CHKERRQ(ierr);
  return 0;
}
//ENDFULLASSEMBLE
