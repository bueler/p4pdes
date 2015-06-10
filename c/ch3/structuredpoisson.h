#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

// Assemble structured-grid Laplacian matrix A, using Dirichlet boundary
// conditions, using a local stencil.  This method forms operator A from
// finite difference approximation of
//     A u = - hx * hy * (u_xx + u_yy)
PetscErrorCode formdirichletlaplacian(DM da, DMDALocalInfo info,
                   PetscReal hx, PetscReal hy, PetscReal diagentry, Mat A);

// For a particular "manufactured" Poisson problem on a square,
// compute the exact solution.
PetscErrorCode formExact(DM da, DMDALocalInfo info, PetscReal hx, PetscReal hy, Vec uexact);

// For a particular "manufactured" Poisson problem on a square,
// compute the right side b of the system  A u = b.
PetscErrorCode formRHS(DM da, DMDALocalInfo info, PetscReal hx, PetscReal hy, Vec b);

#endif

