#include <petscmat.h>
#include <petscdmda.h>

//CREATEMATRIX
PetscErrorCode formdirichletlaplacian(DM da, PetscReal dirichletdiag, Mat A) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  PetscInt   i, j;
  PetscReal  hx = 1./(double)(info.mx-1),  // domain is [0,1] x [0,1]
             hy = 1./(double)(info.my-1);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscReal   v[5];
      PetscInt    ncols = 0;
      row.j = j;               // row of A corresponding to the unknown at (x_i,y_j)
      row.i = i;
      col[ncols].j = j;        // in that diagonal entry ...
      col[ncols].i = i;
      if ( (i==0) || (i==info.mx-1) || (j==0) || (j==info.my-1) ) { // ... on bdry
        v[ncols++] = dirichletdiag;
      } else {
        v[ncols++] = 2*(hy/hx + hx/hy); // ... everywhere else we build a row
        // if neighbor is NOT known to be zero we put an entry:
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

