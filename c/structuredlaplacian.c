#include <petscmat.h>
#include <petscdmda.h>
#include "convenience.h"

//CREATEMATRIX
PetscErrorCode formlaplacian(DM da, Mat A) {
  PetscErrorCode ierr;
  DMDALocalInfo  info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  PetscInt       i, j;
  PetscReal      hx = 1./info.mx,  hy = 1./info.my;  // domain is [0,1]x[0,1]
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      MatStencil  row, col[5];
      PetscReal   v[5];
      PetscInt    ncols = 0;
      row.j        = j; row.i = i;
      col[ncols].j = j; col[ncols].i = i; v[ncols++] = 2*(hx/hy + hy/hx);
      if (i>0)         {col[ncols].j = j;   col[ncols].i = i-1; v[ncols++] = -hy/hx;}
      if (i<info.mx-1) {col[ncols].j = j;   col[ncols].i = i+1; v[ncols++] = -hy/hx;}
      if (j>0)         {col[ncols].j = j-1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      if (j<info.my-1) {col[ncols].j = j+1; col[ncols].i = i;   v[ncols++] = -hx/hy;}
      ierr = MatSetValuesStencil(A,1,&row,ncols,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  matassembly(A)
  return 0;
}
//ENDCREATEMATRIX

