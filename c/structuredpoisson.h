#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

PetscErrorCode formExact(DM da, Vec uexact);

PetscErrorCode formRHS(DM da, Vec b);

// assemble structured-grid Laplacian matrix A, using Dirichlet boundary conditions,
//     using a local stencil
// forms operator A from finite difference approximation of
//     A u = - hx * hy * (u_xx + u_yy)
PetscErrorCode formdirichletlaplacian(DM da, PetscReal dirichletdiag, Mat A);

#endif

