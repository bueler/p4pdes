#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

// For a particular "manufactured" Poisson problem on a square,
// compute the exact solution.
PetscErrorCode formExact(DM da, Vec uexact);

// For a particular "manufactured" Poisson problem on a square,
// compute the right side b of the system  A u = b.
PetscErrorCode formRHS(DM da, Vec b);

// Assemble structured-grid Laplacian matrix A, using Dirichlet boundary
// conditions, using a local stencil.  This method forms operator A from
// finite difference approximation of
//     A u = - hx * hy * (u_xx + u_yy)
PetscErrorCode formdirichletlaplacian(DM da, PetscReal dirichletdiag, Mat A);

#endif

