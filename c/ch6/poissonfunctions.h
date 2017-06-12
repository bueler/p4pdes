#ifndef POISSONFUNCTIONS_H_
#define POISSONFUNCTIONS_H_

/*
These functions promote code reuse and serve as canonical examples.  They
are used in ch6/fish.c, ch6/minimal.c, and ch11/obstacle.c.

The functions FormXDFunctionLocal() compute residuals for the 1D, 2D, 3D
Poisson equation with Dirichlet boundary conditions on a structured grid (DMDA)
on a interval, rectangle, or rectangular solid.  They are designed to be
used as call-backs,

  ierr = DMDASNESSetFunctionLocal(dmda,INSERT_VALUES,
             (DMDASNESFunction)FormXDFunctionLocal,&user); CHKERRQ(ierr);

where X=1,2,3.

The FormXDJacobianLocal() functions assemble Jacobians for the same problems,
and are designed to be call-backs,

  ierr = DMDASNESSetJacobianLocal(dmda,
             (DMDASNESJacobian)FormXDJacobianLocal,&user); CHKERRQ(ierr);

The matrices A are normalized so that A / (hx * hy) approximates the Laplacian.
Thus the entries are O(1).  The entries are integers if hx = hy.
*/

typedef struct {
    // the Dirichlet boundary condition for g(x,y,z); same as exact solution when exists
    double (*g_bdry)(double x, double y, double z);
    // the right-hand-side f(x,y,z) = - laplacian u:
    double (*f_rhs)(double x, double y, double z);
    void   *ctx;  // additional context; see example usage in minimal.c
} PoissonCtx;


PetscErrorCode Form1DFunctionLocal(DMDALocalInfo *info,
    double *au, double *aF, PoissonCtx *user);

PetscErrorCode Form2DFunctionLocal(DMDALocalInfo *info,
    double **au, double **aF, PoissonCtx *user);

PetscErrorCode Form3DFunctionLocal(DMDALocalInfo *info,
    double ***au, double ***aF, PoissonCtx *user);

/* This generates the classical tridiagonal sparse matrix with 2 on the diagonal
and -1 on the off diagonal, if hx = hy.  The boundary conditions correspond to
1 on the diagonal.  For example,
    ch6/fish -fsh_dim 1 -mat_view ::ascii_dense -da_refine 1
produces
 1.00000e+00   0.00000e+00   0.00000e+00   0.00000e+00  0.00000e+00
 0.00000e+00   2.00000e+00  -1.00000e+00   0.00000e+00  0.00000e+00
 0.00000e+00  -1.00000e+00   2.00000e+00  -1.00000e+00  0.00000e+00
 0.00000e+00   0.00000e+00  -1.00000e+00   2.00000e+00  0.00000e+00
 0.00000e+00   0.00000e+00   0.00000e+00   0.00000e+00  1.00000e+00       */
PetscErrorCode Form1DJacobianLocal(DMDALocalInfo *info, PetscScalar *au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* This generates the expected 2m-1 bandwidth (if m=mx=my) sparse matrix with
4 or 1 on the diagonal and -1 or zero in all off-diagonal positions (if hx=hy).
For example,
    ch6/fish -fsh_dim 2 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.                 */
PetscErrorCode Form2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* This generates the expected 2m^2-1 bandwidth (if m=mx=my=mz) sparse matrix
with 6 or 1 on the diagonal and -1 or zero in off-diagonal positions (if hx=hy).
For example,
    ch6/fish -fsh_dim 3 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.
*/
PetscErrorCode Form3DJacobianLocal(DMDALocalInfo *info, PetscScalar ***au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

#endif

