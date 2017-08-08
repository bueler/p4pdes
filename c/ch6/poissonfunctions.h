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

All of these function work with equally-spaced, but independently in each
dimension, structured grids.  That is, the dimensions hx, hy, hz of the
rectangular cells can have any values.

The matrices A are normalized so that if cells are square (h = hx = hy = hz)
then A / h^d approximates the Laplacian in d dimensions.  This is the way
the rows would be scaled in a Galerkin FEM scheme.  (Thus the entries are O(1)
only if d=2.)  The Dirichlet boundary conditions generate diagonal Jacobian
entries with the same values as the diagonal entries for points in
the interior; the Jacobian matrices here have constant diagonal.
*/

//STARTDECLARE
typedef struct {
    // the Dirichlet boundary condition for g(x,y,z)
    double (*g_bdry)(double x, double y, double z, void *ctx);
    // the right-hand-side f(x,y,z) = - laplacian u:
    double (*f_rhs)(double x, double y, double z, void *ctx);
    void   *addctx;  // additional context; see example usage in minimal.c
} PoissonCtx;

PetscErrorCode Form1DFunctionLocal(DMDALocalInfo *info,
    double *au, double *aF, PoissonCtx *user);

PetscErrorCode Form2DFunctionLocal(DMDALocalInfo *info,
    double **au, double **aF, PoissonCtx *user);

PetscErrorCode Form3DFunctionLocal(DMDALocalInfo *info,
    double ***au, double ***aF, PoissonCtx *user);
//ENDDECLARE

/* This generates a tridiagonal sparse matrix.  For example, see the result
of
    ./fish -fsh_dim 1 -mat_view ::ascii_dense -da_refine N                */
PetscErrorCode Form1DJacobianLocal(DMDALocalInfo *info, PetscScalar *au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* If h = hx = hy and h = L/(m-1) then this generates a 2m-1 bandwidth
sparse matrix with 4 on the diagonal and -1 or zero in off-diagonal positions.
For example,
    ./fish -fsh_dim 2 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.                 */
PetscErrorCode Form2DJacobianLocal(DMDALocalInfo *info, PetscScalar **au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* If h = hx = hy = hz and h = L/(m-1) then this generates a 2m^2-1 bandwidth
sparse matrix.  For example,
    ./fish -fsh_dim 3 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.
*/
PetscErrorCode Form3DJacobianLocal(DMDALocalInfo *info, PetscScalar ***au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

#endif

