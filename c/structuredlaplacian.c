#include <petscmat.h>
#include <petscdmda.h>

//CREATEMATRIX
PetscErrorCode formlaplacian(DM da, Mat A) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info); CHKERRQ(ierr);

  PetscInt       i, j;
  PetscReal      hx = 1./info.mx,  hy = 1./info.my;  // domain is [0,1] x [0,1]
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscReal   v[5];
      PetscInt    ncols = 0;
      row.j = j;               // the row of A corresponding to the unknown at
      row.i = i;               //     coordinates (x_i,y_j)
      col[ncols].j = j;        // in the diagonal entry ...
      col[ncols].i = i;
      v[ncols++]   = 2*(hx/hy + hy/hx); // ... we put 4 (if hx = hy)
      if (i>0) {               // except at the x=0 boundary ...
        col[ncols].j = j;
        col[ncols].i = i-1;    //     ... we put -1 (hx = hy case) in the column
        v[ncols++]   = -hy/hx; //     corresponding to (x_{i-1},y_j)
      }
      if (i<info.mx-1) {       // except at the x=1 boundary ...
        col[ncols].j = j;
        col[ncols].i = i+1;
        v[ncols++]   = -hy/hx;
      }
      if (j>0) {               // except at the y=0 boundary ...
        col[ncols].j = j-1;
        col[ncols].i = i;
        v[ncols++]   = -hx/hy;}
      if (j<info.my-1) {       // except at the y=1 boundary ...
        col[ncols].j = j+1;
        col[ncols].i = i;
        v[ncols++]   = -hx/hy;
      }
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
//ENDCREATEMATRIX

