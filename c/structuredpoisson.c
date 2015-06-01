#include <petscmat.h>
#include <petscdmda.h>

//FORMEXACTRHS
PetscErrorCode formExact(DM da, DMDALocalInfo info, PetscReal hx, PetscReal hy, Vec uexact) {
  PetscErrorCode ierr;
  PetscInt       i, j;
  PetscReal      x, y, **auexact;
  ierr = DMDAVecGetArray(da, uexact, &auexact);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * hy;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * hx;
      auexact[j][i] = x*x * (1.0 - x*x) * y*y * (y*y - 1.0);
    }
  }
  ierr = DMDAVecRestoreArray(da, uexact, &auexact);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(uexact); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(uexact); CHKERRQ(ierr);
  return 0;
}

PetscErrorCode formRHS(DM da, DMDALocalInfo info, PetscReal hx, PetscReal hy, Vec b) {
  PetscErrorCode ierr;
  PetscInt       i, j;
  PetscReal      x, y, x2, y2, f, **ab;
  ierr = DMDAVecGetArray(da, b, &ab);CHKERRQ(ierr);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    y = j * hy;  y2 = y*y;
    for (i=info.xs; i<info.xs+info.xm; i++) {
      x = i * hx;  x2 = x*x;
      if ( (i>0) && (i<info.mx-1) && (j>0) && (j<info.my-1) ) { // if not bdry
        // f = - (u_xx + u_yy)  where u is exact
        f = 2.0 * ( (1.0 - 6.0*x2) * y2 * (1.0 - y2)
                    + (1.0 - 6.0*y2) * x2 * (1.0 - x2) );
        ab[j][i] = hx * hy * f;
      } else {
        ab[j][i] = 0.0;                          // on bdry we have 1*u = 0
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &ab); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);
  return 0;
}
//ENDFORMEXACTRHS

//CREATEMATRIX
PetscErrorCode formdirichletlaplacian(DM da, DMDALocalInfo info,
                   PetscReal hx, PetscReal hy, PetscReal dirichletdiag, Mat A) {
  PetscErrorCode ierr;
  PetscInt  i, j;
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscReal   v[5];
      PetscInt    ncols = 0;
      row.j = j;               // row of A corresponding to (x_i,y_j)
      row.i = i;
      col[ncols].j = j;        // in this diagonal entry
      col[ncols].i = i;
      if ( (i==0) || (i==info.mx-1) || (j==0) || (j==info.my-1) ) {
        // if on boundary, just insert diagonal entry
        v[ncols++] = dirichletdiag;
      } else {
        v[ncols++] = 2*(hy/hx + hx/hy); // ... everywhere else we build a row
        // if neighbor is NOT a known boundary value then we put an entry:
        if (i-1>0) {
          col[ncols].j = j;    col[ncols].i = i-1;  v[ncols++] = -hy/hx;  }
        if (i+1<info.mx-1) {
          col[ncols].j = j;    col[ncols].i = i+1;  v[ncols++] = -hy/hx;  }
        if (j-1>0) {
          col[ncols].j = j-1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
        if (j+1<info.my-1) {
          col[ncols].j = j+1;  col[ncols].i = i;    v[ncols++] = -hx/hy;  }
      }
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
//ENDCREATEMATRIX

