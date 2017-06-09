#ifndef JACOBIANS_H_
#define JACOBIANS_H_

/*
These functions promote code reuse and serve as canonical examples.  They
are used in all the codes in ch6/ and in ch11/obstacle.c

They assemble Jacobians for the 1D, 2D, 3D Poisson equation with
Dirichlet boundary conditions on structured grids (DMDA) on the unit square.
That is, these are discretized Laplacian operators.  They are designed to be
used as call-backs,

  ierr = DMDASNESSetJacobianLocal(dmda,
             (DMDASNESJacobian)FormXDJacobianLocal,&user); CHKERRQ(ierr);

where X=1,2,3.

The matrices A are normalized so that A / (hx * hy) approximates the Laplacian.
Thus the entries are O(1), and the entries are integers if hx = hy.
*/

/* For example,
    ch6/fish1 -mat_view ::ascii_dense -da_refine 1
produces
 1.00000e+00   0.00000e+00   0.00000e+00   0.00000e+00  0.00000e+00
 0.00000e+00   2.00000e+00  -1.00000e+00   0.00000e+00  0.00000e+00
 0.00000e+00  -1.00000e+00   2.00000e+00  -1.00000e+00  0.00000e+00
 0.00000e+00   0.00000e+00  -1.00000e+00   2.00000e+00  0.00000e+00
 0.00000e+00   0.00000e+00   0.00000e+00   0.00000e+00  1.00000e+00
*/
PetscErrorCode Form1DJacobianLocal(DMDALocalInfo *info, PetscScalar *au,
                                   Mat J, Mat Jpre, void *user);

/* For example,
    ch6/fish2 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave and which has
1 or 4 on diagonal, with 1 only for the Dirichlet boundary locations,
and -1 or 0 in the off-diagonal positions.
*/
PetscErrorCode Form2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                   Mat J, Mat Jpre, void *user);

/* For example,
    ch6/fish3 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave and which has
1 or 6 on diagonal, with 1 only for the Dirichlet boundary locations,
and -1 or 0 in the off-diagonal positions.
*/
PetscErrorCode Form3DJacobianLocal(DMDALocalInfo *info, PetscScalar ***au,
                                   Mat J, Mat Jpre, void *user);

#endif

