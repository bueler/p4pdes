#ifndef STRUCTUREDLAPLACIAN_H_
#define STRUCTUREDLAPLACIAN_H_

// assemble structured-grid Laplacian matrix A, using Dirichlet boundary conditions,
//   using a local stencil;  forms A in  A u = - hx * hy * (u_xx + u_yy)
PetscErrorCode formdirichletlaplacian(DM da, Mat A);

#endif

