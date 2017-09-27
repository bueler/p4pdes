#ifndef POISSONFUNCTIONS_H_
#define POISSONFUNCTIONS_H_

/*
These functions approximate the residual of, and the Jacobian of, the
(slightly-generalized) Poisson equation
    - cx u_xx - cy u_yy - cz u_zz = f(x,y)
with Dirichlet boundary conditions  u = g(x,y).

These functions promote code reuse and serve as canonical examples.  They
are used in ch6/fish.c, ch6/minimal.c, and ch12/obstacle.c.

The functions FormXDFunctionLocal() compute residuals for the 1D, 2D, 3D
problems on a structured grid (DMDA).  The domain is an interval, rectangle,
or rectangular solid.  These functions are designed to be used as call-backs:

  ierr = DMDASNESSetFunctionLocal(dmda,INSERT_VALUES,
             (DMDASNESFunction)FormXDFunctionLocal,&user); CHKERRQ(ierr);

where X=1,2,3.

The FormXDJacobianLocal() functions are call-backs which assemble Jacobians
for the same problems:

  ierr = DMDASNESSetJacobianLocal(dmda,
             (DMDASNESJacobian)FormXDJacobianLocal,&user); CHKERRQ(ierr);

All of these function work with equally-spaced structured grids.  The
dimensions hx, hy, hz of the rectangular cells can have any positive values.

The matrices A are normalized so that if cells are square (h = hx = hy = hz)
then A / h^d approximates the Laplacian in d dimensions.  This is the way
the rows would be scaled in a Galerkin FEM scheme.  (The entries are O(1)
only if d=2.)

The Dirichlet boundary conditions are approximated using diagonal Jacobian
entries with the same values as the diagonal entries for points in
the interior.  Thus these Jacobian matrices have constant diagonal.
*/

// warning: the user is in charge of setting up ALL of this content!
//STARTDECLARE
typedef struct {
    // the coefficients in  - cx u_xx - cy u_yy - cz u_zz = f
    double cx, cy, cz;
    // the right-hand-side f(x,y,z)
    double (*f_rhs)(double x, double y, double z, void *ctx);
    // the Dirichlet boundary condition g(x,y,z)
    double (*g_bdry)(double x, double y, double z, void *ctx);
    void   *addctx;  // additional context; see example usage in minimal.c
} PoissonCtx;

PetscErrorCode Form1DFunctionLocal(DMDALocalInfo *info,
    double *au, double *aF, PoissonCtx *user);

PetscErrorCode Form2DFunctionLocal(DMDALocalInfo *info,
    double **au, double **aF, PoissonCtx *user);

PetscErrorCode Form3DFunctionLocal(DMDALocalInfo *info,
    double ***au, double ***aF, PoissonCtx *user);
//ENDDECLARE

/* This generates a tridiagonal sparse matrix.  If cx=1 then it has 2 on the
diagonal and -1 or zero in off-diagonal positions.  For example,
    ./fish -fsh_dim 1 -mat_view ::ascii_dense -da_refine N                */
PetscErrorCode Form1DJacobianLocal(DMDALocalInfo *info, double *au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* If h = hx = hy and h = L/(m-1) then this generates a 2m-1 bandwidth
sparse matrix.  If cx=cy=1 then it has 4 on the diagonal and -1 or zero in
off-diagonal positions.  For example,
    ./fish -fsh_dim 2 -mat_view :foo.m:ascii_matlab -da_refine N
produces a matrix which can be read into Matlab/Octave.                   */
PetscErrorCode Form2DJacobianLocal(DMDALocalInfo *info, double **au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* If h = hx = hy = hz and h = L/(m-1) then this generates a 2m^2-1 bandwidth
sparse matrix.  For example,
    ./fish -fsh_dim 3 -mat_view :foo.m:ascii_matlab -da_refine N          */
PetscErrorCode Form3DJacobianLocal(DMDALocalInfo *info, double ***au,
                                   Mat J, Mat Jpre, PoissonCtx *user);

/* The following function generates an initial iterate using either
  * an interpolant of the boundary function g (reasonably smooth)
  * a random function (no smoothness)
  * zero
In addition, one can initialize either using the boundary function g for
the boundary locations in the initial state, or not.                      */

typedef enum {GINTERPOLANT, RANDOM, ZEROS} InitialType;

PetscErrorCode InitialState(DMDALocalInfo *info, InitialType it, PetscBool gbdry,
                            Vec u, PoissonCtx *user);

#endif

